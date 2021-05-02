# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from math import e


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.binary_cross_entropy_with_logits(input, target, reduction='none', weight=self.weight),
                          self.gamma)


class CSRLoss(nn.Module):
    def __init__(self):
        super(CSRLoss, self).__init__()

    def forward(self, logits, labels):
        max_true_pos = torch.max(logits * labels, axis=1)[0]
        max_true_neg = torch.max(logits * (1 - labels), axis=1)[0]
        return torch.mean(
            torch.max(0 * max_true_pos, 0.5 + 0.5 - torch.sigmoid(max_true_pos)) + torch.max(0 * max_true_neg,
                                                                                             0.5 + torch.sigmoid(
                                                                                                 max_true_neg) - 0.5))


class LDAMLossOLD(nn.Module):
    def __init__(self, class_counts=None, C=1, max_m=0.5):
        super(LDAMLoss, self).__init__()
        self.class_counts = class_counts
        self.C = C
        self.max_m = max_m

    def compute_qx_og(self, logits, labels):
        logits = torch.sigmoid(logits)
        assert self.class_counts.shape[-1] == logits.shape[-1] == labels.shape[
            -1], "Mismatch in counts/logits/labels shapes"

        true_pos_scores = logits * labels
        true_pos_scores = true_pos_scores[true_pos_scores != 0]
        # true_pos_scores = true_pos_scores.reshape(logits.shape[0], -1)

        true_neg_scores = logits * (1 - labels)
        # true_neg_scores = true_neg_scores[true_neg_scores != 0]
        # true_neg_scores = true_neg_scores.reshape(logits.shape[0], -1)

        # m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        # m_list = m_list * (max_m / np.max(m_list))

        delta_y = (self.C / (self.class_counts ** .25)) * labels
        delta_y = delta_y * (self.max_m / torch.max(delta_y))
        delta_y = delta_y[delta_y != 0]

        numerator = e ** (true_pos_scores - delta_y)
        denominator = e ** (true_pos_scores - delta_y)

        slices = torch.sum(labels, axis=1)  # [2, 5, 6]
        true_neg_scores = torch.sum(true_neg_scores, axis=-1)

        slice_counter = 0
        for slice_ind, to_add in zip(slices, true_neg_scores):
            slice_counter += int(slice_ind.item())
            last = slice_counter - int(slice_ind.item())
            denominator = torch.cat(
                (denominator[:last], denominator[last:slice_counter] + to_add.item(), denominator[slice_counter:]))

        return numerator / denominator

    def compute_qx(self, logits, labels):
        logits = torch.sigmoid(logits)
        assert self.class_counts.shape[-1] == logits.shape[-1] == labels.shape[
            -1], "Mismatch in counts/logits/labels shapes"

        # logits: [.99, .001, .54, .4]
        # labesl: [  1,    0,   0,  1]

        true_pos_scores = logits * labels  # [0.99, 0, 0, 4]
        true_pos_scores = true_pos_scores[true_pos_scores != 0]  # [0.99, 4]
        # true_pos_scores = true_pos_scores.reshape(logits.shape[0], -1)

        true_neg_scores = logits * (1 - labels)  # [0, .001, .54, 0]
        # true_neg_scores = true_neg_scores[true_neg_scores != 0]
        # true_neg_scores = true_neg_scores.reshape(logits.shape[0], -1)

        # m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        # m_list = m_list * (max_m / np.max(m_list))

        delta_y = (self.C / (self.class_counts ** .25)) * labels
        delta_y = delta_y * (self.max_m / torch.max(delta_y))
        delta_y = delta_y[delta_y != 0]

        numerator = e ** (true_pos_scores - delta_y)
        denominator = e ** (true_pos_scores - delta_y)

        slices = torch.sum(labels, axis=1)  # [2, 5, 6]
        true_neg_scores = torch.sum(true_neg_scores, axis=-1)

        sum_all_scores = true_neg_scores + torch.sum(true_pos_scores, axis=-1)
        # sum_all_scores *= torch.ones(true_pos_scores.shape)
        # sum_all_scores -= true_pos_scores
        # print('sum_all_scores:'.upper(),sum_all_scores)

        slice_counter = 0
        for slice_ind, to_add in zip(slices, sum_all_scores):
            slice_counter += int(slice_ind.item())
            last = slice_counter - int(slice_ind.item())
            denominator = torch.cat(
                (denominator[:last], denominator[last:slice_counter] + to_add.item(), denominator[slice_counter:]))

        denominator -= true_pos_scores

        return numerator / denominator

    def forward(self, logits, labels):

        loss = 0

        part_one = torch.log(self.compute_qx(logits, labels))

        part_two = torch.log(1 - self.compute_qx(logits, 1 - labels))

        loss = (torch.mean(-part_one) + torch.mean(-part_two))
        return loss


class BalancedBCEWithLogitsLoss(nn.Module):

    def __init__(self, grad_clip=False, weights=None, reduction='mean'):
        super(BalancedBCEWithLogitsLoss, self).__init__()
        self.grad_clip = grad_clip
        self.weights = weights
        self.reduction = reduction

    def forward(self, logits, labels):
        assert logits.shape == labels.shape, "logits shape %r != labels shape %r" % (logits.shape, labels.shape)

        # number of classes
        nc = labels.shape[1]

        # number of positive classes per example in batch
        npos_per_example = labels.sum(1)  # shape: [batch_size]

        # alpha: ratio of negative classes per example in batch
        alpha = (nc - npos_per_example) / npos_per_example
        alpha[alpha == float("Inf")] = 0
        alpha = alpha.unsqueeze(1).expand_as(labels)  # shape: [batch_size, num_classes]

        # positive weights
        pos_weight = labels * alpha

        # to avoid gradients vanishing
        if self.grad_clip:
            logits = logits.clamp(min=-14.0, max=14.0)

        proba = torch.sigmoid(logits)
        # see https://pytorch.org/docs/stable/nn.html#torch.nn.BCEWithLogitsLoss for loss eq.
        loss = -(torch.log(proba) * pos_weight + torch.log(1. - proba) * (1. - labels))
        # the labels which are supposed to be positive get more weight added to them
        loss = loss.mean()
        return loss


class LDAMLoss(nn.Module):
    def __init__(self, class_counts=None, C=1, max_m=0.5, grad_clip=True):
        super(LDAMLoss, self).__init__()
        self.class_counts = class_counts
        self.C = C
        self.max_m = max_m
        self.grad_clip = grad_clip

    def compute_qx(self, logits, labels):
        """

        :param logits: shape:
        :param labels:
        :return:
        """
        assert logits.shape == labels.shape, print("Shape mismatch\n", logits, '\n', labels)
        assert labels.dtype == torch.float
        assert self.class_counts.dtype == torch.float
        assert self.class_counts.shape[-1] == logits.shape[-1] == labels.shape[
            -1], "Mismatch in counts/logits/labels shapes"

        if self.grad_clip:
            logits = logits.clamp(min=-100.0, max=100.0)

        # logits = nn.Softmax(dim=-1)(logits)
        true_pos_indices = torch.nonzero(labels, as_tuple=True)  # indices of positive labels
        true_pos_scores = logits[true_pos_indices]

        true_neg_indices = torch.nonzero((1 - labels), as_tuple=True)
        true_neg_scores = logits[true_neg_indices]

        # m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        # m_list = m_list * (max_m / np.max(m_list))

        delta_y = (self.C / (self.class_counts ** .25)) * labels
        delta_y = delta_y * (self.max_m / torch.max(delta_y))
        delta_y = delta_y[delta_y != 0]

        # softmax plus margin subtraction
        numerator = e ** (true_pos_scores - delta_y)
        denominator = e ** (true_pos_scores - delta_y) + torch.sum(e ** true_neg_scores, axis=-1)
        return numerator / denominator


    def forward(self, logits, labels):
        if self.grad_clip:
            logits = logits.clamp(min=-14.0, max=14.0)
        loss = -torch.log(self.compute_qx(logits, labels))
        loss = loss.mean()
        return loss


# class LDAMLoss(nn.Module):
#
#     def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
#         super(LDAMLoss, self).__init__()
#         m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
#         m_list = m_list * (max_m / np.max(m_list))
#         m_list = torch.cuda.FloatTensor(m_list)
#         self.m_list = m_list
#         assert s > 0
#         self.s = s
#         self.weight = weight
#
#     def forward(self, x, target):
#         index = torch.zeros_like(x, dtype=torch.uint8)
#         index.scatter_(1, target.data.view(-1, 1), 1)
#
#         index_float = index.type(torch.cuda.FloatTensor)
#         batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
#         batch_m = batch_m.view((-1, 1))
#         x_m = x - batch_m
#
#         output = torch.where(index, x_m, x)
#         return F.cross_entropy(self.s * output, target, weight=self.weight)


# class MultiTaskLossWrapper(nn.Module):
#     def __init__(self, task_num):
#         super(MultiTaskLossWrapper, self).__init__()
#         self.task_num = task_num
#         self.log_vars = nn.Parameter(torch.zeros((task_num)))
#
#     def forward(self, loss, idx):
#         prec = torch.exp(-self.log_vars[idx])
#         loss = prec * loss + self.log_vars[idx]
#         return loss

# class MultiTaskLossWrapper(nn.Module):
#     def __init__(self, task_num):
#         super(MultiTaskLossWrapper, self).__init__()
#         self.task_num = task_num
#         self.log_vars = nn.Parameter(torch.zeros((task_num)))
#
#     def forward(self, loss, idx):
#         prec = self.log_vars[idx]
#         loss = prec * loss
#         return loss


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.ones((task_num)))

    def forward(self, mlcc_loss, mcc_loss):
        mlcc_loss = 1 / (2 * self.log_vars[0]**2) * mlcc_loss
        mcc_loss = 1 / (2 * self.log_vars[1]**2) * mcc_loss
        loss = mlcc_loss + mcc_loss + torch.log(self.log_vars[0]*self.log_vars[1])
        return loss, mlcc_loss, mcc_loss


# class DynamicWeightAverager(nn.Module):
#     def __init__(self, task_num):
#         super(MultiTaskLossWrapper, self).__init__()
#         self.task_num = task_num
#         self.L_t_minus_1 = []
#         self.L_t_minus_2 = []
#
#     def forward(self, mlcc_loss, mcc_loss):
#
#         w_mlcc = 1
#
#
#
#
#
#         return loss, mlcc_loss, mcc_loss
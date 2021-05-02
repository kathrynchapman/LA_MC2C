import pandas as pd
import sys

# sys.path.insert(1, '../CLEF_Datasets_ICD/processed_data/')
from process_data import *
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.nn import Linear
import io
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import logging
import random
import json
import argparse
from loss import *
import random
from utils import *
from RAkEL import *
from label_clusterer import *
from collections import defaultdict
import scipy.stats as ss
from torch.utils.data import Dataset, RandomSampler, DataLoader, SequentialSampler

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from transformers.modeling_bert import BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_xlm_roberta import XLMRobertaModel
from transformers.modeling_xlm_roberta import XLMRobertaModel, XLMRobertaConfig

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          XLMRobertaConfig, XLMRobertaForSequenceClassification,
                          XLMRobertaTokenizer, AdamW, get_linear_schedule_with_warmup)


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, args='', loss_fct='', class_weights=None):
        super().__init__(config)
        self.args = args
        if self.args.encoder_type == 'bert':
            self.bert = BertModel(config)
        elif self.args.encoder_type == 'xlmroberta':
            self.roberta = XLMRobertaModel(config)
        self.num_labels = args.num_labels
        if not class_weights:
            self.class_weights = torch.ones((self.num_labels,))
        else:
            self.class_weights = class_weights
        self.iteration = 1
        self.loss_fct = loss_fct
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def freeze_bert_encoder(self):
        if self.args.n_bert2freeze:
            for name, param in self.bert.named_parameters():
                if name.split('.')[1] == 'encoder':
                    layer_n = name.split('.')[3]
                    if int(layer_n) < self.args.n_bert2freeze:
                        param.requires_grad = False
        else:
            for param in self.bert.parameters():
                param.requires_grad = False


    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
            self,
            doc_input_ids=None,
            doc_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            ranks=None,
            output_attentions=None,
            epoch=1,
            doc_ids=None,
            train=False,
            t_total=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import BertTokenizer, BertForSequenceClassification
        import torch
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
        """
        if self.args.encoder_type == 'xlmroberta':
            outputs = self.roberta(
                doc_input_ids,
                attention_mask=doc_attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                # output_attentions=output_attentions,
            )
        elif self.args.encoder_type == 'bert':
            outputs = self.bert(
                doc_input_ids,
                attention_mask=doc_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                # output_attentions=output_attentions,
            )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        logits = logits.view(-1, self.num_labels)

        if self.args.doc_batching:
            if self.args.logit_aggregation == 'max':
                logits = torch.max(logits, axis=0)[0]
            elif self.args.logit_aggregation == 'avg':
                logits = torch.mean(logits, axis=0)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.args.do_iterative_class_weights:
                # temp = logits.view(-1, self.num_labels) - labels.view(-1, self.num_labels) + 1
                temp = logits.detach()
                temp = torch.nn.Sigmoid()(temp)
                temp = (temp > self.args.prediction_threshold).float()
                temp = torch.mean(
                    torch.abs(temp.view(-1, self.num_labels) - labels.view(-1, self.num_labels)).float() + 1, axis=0)
                try:
                    self.class_weights = torch.Tensor(self.class_weights).cuda()
                except:
                    pass
                self.class_weights *= self.iteration
                self.class_weights += temp
                self.class_weights /= (self.iteration + 1)
                class_weights = self.class_weights.detach()
            elif self.args.DRW:
                idx = epoch // 160
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], self.class_weights)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.class_weights)
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            else:
                class_weights = torch.Tensor(self.class_weights).cuda()

            labels = labels.float()

            if self.loss_fct == 'bce':
                if self.args.use_bce_class_weights:
                    loss_fct = BCEWithLogitsLoss(pos_weight=class_weights)
                else:
                    loss_fct = BCEWithLogitsLoss()
            elif self.loss_fct == 'bbce':
                loss_fct = BalancedBCEWithLogitsLoss(grad_clip=True, weights=class_weights)
            elif self.loss_fct == 'ldam':
                loss_fct = LDAMLoss(class_weights, max_m=self.args.max_m)
            elif self.loss_fct == 'focal':
                loss_fct = FocalLoss(weight=per_cls_weights)
            elif self.loss_fct == 'csr':
                loss_fct = CSRLoss()

            if self.loss_fct != 'none':
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            else:
                loss = 0

            outputs = (loss,) + outputs

        self.iteration += 1
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForMLSCWithLabelAttention(BertPreTrainedModel):
    def __init__(self, config, args='', loss_fct='', class_weights=None):
        super().__init__(config)
        self.args = args
        if self.args.encoder_type == 'bert':
            self.bert = BertModel(config)
        elif self.args.encoder_type == 'xlmroberta':
            self.roberta = XLMRobertaModel(config)
        self.num_labels = args.num_labels
        self.label_data = ''
        if not class_weights:
            self.class_weights = torch.ones((self.num_labels,))
        else:
            self.class_weights = class_weights
        self.iteration = 1
        self.loss_fct = loss_fct
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.w1 = torch.nn.Linear(args.doc_max_seq_length, 1)
        self.relu = nn.ReLU()
        self.w2 = torch.nn.Linear(args.label_max_seq_length, 1)
        self.hidden_size = config.hidden_size
        self.init_weights()

    def initialize_label_data(self, label_data):
        self.label_data = label_data

    def freeze_bert_encoder(self):
        if self.args.n_bert2freeze:
            for name, param in self.bert.named_parameters():
                if name.split('.')[1] == 'encoder':
                    layer_n = name.split('.')[3]
                    if int(layer_n) < self.args.n_bert2freeze:
                        param.requires_grad = False
        else:
            for param in self.bert.parameters():
                param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
            self,
            doc_input_ids=None,
            doc_attention_mask=None,
            label_desc_input_ids=None,
            label_desc_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            ranks=None,
            output_attentions=None,
            epoch=1,
            doc_ids=None,
            train=False,
            t_total=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import BertTokenizer, BertForSequenceClassification
        import torch
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
        """

        if self.args.encoder_type == 'xlmroberta':
            doc_outputs = self.roberta(
                doc_input_ids,
                attention_mask=doc_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                # output_attentions=output_attentions,
            )

            label_outputs = self.roberta(
                self.label_data[0].cuda(),
                attention_mask=self.label_data[1].cuda(),
                # token_type_ids=token_type_ids,
                # position_ids=position_ids,
                # head_mask=head_mask,
                # inputs_embeds=inputs_embeds,
                # output_attentions=output_attentions,
            )
        elif self.args.encoder_type == 'bert':
            doc_outputs = self.bert(
                doc_input_ids,
                attention_mask=doc_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                # output_attentions=output_attentions,
            )

            label_outputs = self.bert(
                self.label_data[0].cuda(),
                attention_mask=self.label_data[1].cuda(),
                token_type_ids=self.label_data[-1].cuda(),
            )

        # get the sequence-level document representations
        doc_seq_output = doc_outputs[0]

        doc_seq_output = self.dropout(doc_seq_output)

        batch_size = doc_seq_output.shape[0]

        # get the sequence-level label description representations
        label_seq_output = label_outputs[0]

        label_seq_output = label_seq_output.reshape(self.num_labels * self.args.label_max_seq_length, self.hidden_size)
        temp = torch.matmul(doc_seq_output, label_seq_output.T)
        temp = temp.permute(0, 2, 1)

        temp = self.w1(temp)

        # temp = self.relu(temp)

        temp = temp.reshape(batch_size, self.num_labels, self.args.label_max_seq_length)

        temp = self.w2(temp)

        logits = temp.view(-1, self.num_labels)

        if self.args.doc_batching:
            if self.args.logit_aggregation == 'max':
                logits = torch.max(logits, axis=0)[0]
            elif self.args.logit_aggregation == 'avg':
                logits = torch.mean(logits, axis=0)

        # print("logits.shape:", logits.shape)

        outputs = (logits,)  # + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # labels = labels[0,:]
            if self.args.do_iterative_class_weights:
                # temp = logits.view(-1, self.num_labels) - labels.view(-1, self.num_labels) + 1
                temp = logits.detach()
                temp = torch.nn.Sigmoid()(temp)
                temp = (temp > self.args.prediction_threshold).float()
                temp = torch.mean(
                    torch.abs(temp.view(-1, self.num_labels) - labels.view(-1, self.num_labels)).float() + 1, axis=0)
                try:
                    self.class_weights = torch.Tensor(self.class_weights).cuda()
                except:
                    pass
                self.class_weights *= self.iteration
                self.class_weights += temp
                self.class_weights /= (self.iteration + 1)
                class_weights = self.class_weights.detach()
            elif self.args.DRW:
                idx = epoch // 160
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], self.class_weights)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.class_weights)
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            else:
                class_weights = torch.Tensor(self.class_weights).cuda()

            labels = labels.float()

            if self.loss_fct == 'bce':
                if self.args.use_bce_class_weights:
                    loss_fct = BCEWithLogitsLoss(pos_weight=class_weights)
                else:
                    loss_fct = BCEWithLogitsLoss()
            elif self.loss_fct == 'bbce':
                loss_fct = BalancedBCEWithLogitsLoss(grad_clip=True, weights=class_weights)
            elif self.loss_fct == 'ldam':
                loss_fct = LDAMLoss(class_weights, max_m=self.args.max_m)
            elif self.loss_fct == 'focal':
                loss_fct = FocalLoss()
            elif self.loss_fct == 'csr':
                loss_fct = CSRLoss()

            if self.loss_fct != 'none':
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            else:
                loss = 0

            outputs = (loss,) + outputs

            # loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # outputs = (loss,) + outputs

        self.iteration += 1
        return outputs  # (loss), logits, (hidden_states), (attentions)


class MC2C(BertPreTrainedModel):
    def __init__(self, config, args='', loss_fct='', class_weights=None, doc_ids2_clusters=None, clusters=None,
                 cluster_idx2seed=None, cluster_output_dir=None):
        super().__init__(config)
        self.args = args
        self.config = config
        if self.args.encoder_type == 'bert':
            self.bert = BertModel(config)
        elif self.args.encoder_type == 'xlmroberta':
            self.roberta = XLMRobertaModel(config)
        self.num_labels = args.num_labels
        if not class_weights:
            self.class_weights = torch.ones((self.num_labels,))
        else:
            self.class_weights = class_weights
        self.iteration = 1
        self.loss_fct = loss_fct
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.n_clusters = len(clusters)
        self.clusters = clusters
        self.MLCC = torch.nn.Linear(config.hidden_size, self.n_clusters)
        self.doc_ids2_clusters = doc_ids2_clusters
        self.cluster_idx2seed = cluster_idx2seed
        self.cluster_output_dir = cluster_output_dir
        self.create_MCC_classifiers()
        self.init_weights()
        self.cluster_activator_f1 = []
        self.cluster_activator_p = []
        self.cluster_activator_r = []
        self.step = 0
        # self.mtl_loss_combiner = MultiTaskLossWrapper(2)

    def create_MCC_classifiers(self):
        """
        Dynamically generates appropriate number of inner-cluster classifiers, of the right size according to the
        number of labels in a given cluster. Allows for smooth experimenting with different numbers of clusters/
        different cluster sizes
        :return:
        """
        in_size = self.config.hidden_size if not self.args.pass_mlcc_preds_to_mccs else \
            self.config.hidden_size + self.n_clusters
        classifiers = {"self.mcc{}".format(str(i)): torch.nn.Linear(in_size,
                                                                    len(self.clusters[self.cluster_idx2seed[i]]))
                       for i in range(self.n_clusters)}
        for k, v in classifiers.items():
            exec("%s=%s" % (k, v))

    def get_idx2id(self, idx2id):
        self.idx2id = idx2id

    def load_local_labels(self, seed):
        return pickle.load(open(os.path.join(self.cluster_output_dir,'{}/train_doc_id2gold.p'.format(seed)), 'rb'))

    def load_local2global_idx(self, seed):
        return pickle.load(open(os.path.join(self.cluster_output_dir,
                                             '{}/cluster_idx2overall_idx.p'.format(seed)), 'rb'))

    def load_class_counts(self, local_or_global):
        if local_or_global == 'local':
            return pickle.load(open(os.path.join(self.cluster_output_dir, 'local_class_counts.p'), 'rb'))
        elif local_or_global == 'global':
            return pickle.load(open(os.path.join(self.cluster_output_dir, 'global_cluster_counts.p'), 'rb'))

    def convert_from_micro_to_global_logits(self, micro_logit_dict):
        global_logits = dict()
        for doc_id in micro_logit_dict.keys():
            pos_indices = []
            for cls_idx, logits in micro_logit_dict[doc_id].items():
                seed = self.cluster_idx2seed[cls_idx]
                local2global_dict = self.load_local2global_idx(seed)
                logits = nn.Softmax(dim=-1)(logits)
                label = torch.argmax(logits)
                pos_indices.append(local2global_dict[label.item()])

            logits = torch.zeros(self.num_labels)
            logits[pos_indices] = 1
            global_logits[doc_id] = logits
        return global_logits

    def freeze_bert_encoder(self):
        if self.args.n_bert2freeze:
            for name, param in self.bert.named_parameters():
                if name.split('.')[1] == 'encoder':
                    layer_n = name.split('.')[3]
                    if int(layer_n) < self.args.n_bert2freeze:
                        param.requires_grad = False
        else:
            for param in self.bert.parameters():
                param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
            self,
            doc_input_ids=None,
            doc_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            activated_clusters=None,
            labels=None,
            cluster_labels=None,
            local_labels=None,
            ranks=None,
            output_attentions=None,
            epoch=1,
            doc_ids=None,
            train=False,
            debug=False,
            t_total=0,
    ):
        self.step += 1
        if self.args.encoder_type == 'xlmroberta':
            outputs = self.roberta(
                doc_input_ids,
                attention_mask=doc_attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                # output_attentions=output_attentions,
            )
        elif self.args.encoder_type == 'bert':
            outputs = self.bert(
                doc_input_ids,
                attention_mask=doc_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                # output_attentions=output_attentions,
            )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        ###############################################################################################################
        #                          FIRST CLASSIFIER TO PREDICT WHICH CLUSTERS ARE ACTIVATED                           #
        ###############################################################################################################
        # now we have our CLS vector
        cluster_logits = self.MLCC(pooled_output)  # batch_size, n_clusters

        batch_size = cluster_logits.shape[0]

        ids = [self.idx2id[i.item()] for i in doc_ids]
        if train or self.args.eval_cluster_activator:
            cluster_labels = torch.Tensor([self.doc_ids2_clusters[d] for d in ids]).cuda()
            cluster_weights = torch.Tensor(self.load_class_counts('global')).cuda()
            if self.args.loss_fct == 'bce':
                if self.args.use_bce_class_weights:
                    cluster_loss = BCEWithLogitsLoss(pos_weight=cluster_weights)(cluster_logits, cluster_labels)
                else:
                    cluster_loss = BCEWithLogitsLoss()(cluster_logits, cluster_labels)
            elif self.args.loss_fct == 'bbce':
                cluster_loss = BalancedBCEWithLogitsLoss(grad_clip=True)(cluster_logits, cluster_labels)
            else:
                raise Exception(f'Problem with cluster loss: {self.args.loss_fct}')
            if self.args.eval_cluster_activator:
                tmp = torch.nn.Sigmoid()(cluster_logits)
                tmp = (tmp > 0.5)
                f1 = f1_score(y_true=cluster_labels.cpu(), y_pred=tmp.cpu(), average=self.args.metric_avg)
                p = precision_score(y_true=cluster_labels.cpu(), y_pred=tmp.cpu(), average=self.args.metric_avg)
                r = recall_score(y_true=cluster_labels.cpu(), y_pred=tmp.cpu(), average=self.args.metric_avg)
                self.cluster_activator_f1.append(f1)
                self.cluster_activator_p.append(p)
                self.cluster_activator_r.append(r)

        ###############################################################################################################
        ###############################################################################################################

        ###############################################################################################################
        #                          SECOND CLASSIFIER TO PREDICT LABELS WITHIN THE CLUSTERS                            #
        ###############################################################################################################
        if not train:
            # get 'cluster_labels' from cluster logits
            cluster_labels = torch.zeros(cluster_logits.shape)
            cluster_labels_idx = torch.nonzero((torch.nn.Sigmoid()(cluster_logits) > 0.5).int(), as_tuple=True)
            cluster_labels[cluster_labels_idx] = 1



        micro_logit_dict = defaultdict(dict)
        # now we have the indices which tell us which classifiers we need to use
        # the row index --> example, columun index --> relevant classifier for that

        local_counts = self.load_class_counts('local')

        micro_loss_avg = 0
        micro_loss_count = 1

        # micro_loss

        cluster_labels2 = cluster_labels.clone().detach()


        tmp_docids = []
        for ex_idx, (ex_labels, doc_id) in enumerate(zip(cluster_labels2, doc_ids)):
            doc_id = self.idx2id[doc_id.item()]
            tmp_docids.append(doc_id)
            if self.args.train_with_none and train:  # activates all classifiers
                ex_labels += 1
            relevant_classifiers = torch.nonzero(ex_labels)
            for cls_idx in relevant_classifiers:
                cls_idx = cls_idx.item()
                counts = torch.Tensor(
                    local_counts[cls_idx]).cuda()  # gives us the counts for the classes in that cluster
                classifier = getattr(self, 'mcc{}'.format(cls_idx)).cuda()  # call the relevant classifier
                micro_labels = self.load_local_labels(self.cluster_idx2seed[cls_idx])
                if self.args.pass_mlcc_preds_to_mccs:
                    if self.step <= t_total * .1:
                        mcc_input = torch.cat((pooled_output[ex_idx, :], torch.nn.Sigmoid()(cluster_labels2[ex_idx, :].cuda())))
                    else:
                        mcc_input = torch.cat((pooled_output[ex_idx, :], torch.nn.Sigmoid()(cluster_logits[ex_idx, :].cuda())))
                else:
                    mcc_input = pooled_output[ex_idx, :]

                micro_logits = classifier(mcc_input)
                if train:
                    micro_labels = torch.Tensor(micro_labels[doc_id]).cuda()
                    if micro_logits.shape[-1] == 1:
                        micro_loss_fct = BCEWithLogitsLoss(
                            pos_weight=torch.Tensor([(self.args.n_examples - c) / c for c in counts]).cuda())
                        micro_loss = micro_loss_fct(micro_logits.view(-1, len(counts)),
                                                    micro_labels.view(-1, len(counts)))
                    elif self.args.mcc_loss == 'ldam':
                        micro_loss_fct = LDAMLoss(counts, max_m=self.args.max_m, grad_clip=False)
                        # micro_loss_fct = LDAMLoss(counts, max_m=self.args.max_m, grad_clip=True)
                        micro_loss = micro_loss_fct(micro_logits.view(-1, len(counts)),
                                                    micro_labels.view(-1, len(counts)))
                    elif self.args.mcc_loss == 'ce':
                        counts = torch.Tensor([(self.args.n_examples - c) / c for c in counts]).cuda()
                        if self.args.use_mcc_class_weights:
                            micro_loss_fct = CrossEntropyLoss(weight=counts)
                        else:
                            micro_loss_fct = CrossEntropyLoss()
                        micro_loss = micro_loss_fct(micro_logits.view(-1, len(counts)),
                                                    torch.nonzero(micro_labels, as_tuple=True)[0].cuda())
                    if self.args.mlcc_as_gates:
                        # we can use the MLCC predictions as weights for the different MCC losses (gates)
                        gate = torch.nn.Sigmoid()(cluster_logits[ex_idx, cls_idx])
                        micro_loss *= gate
                    # micro_loss = self.mtl_loss_combiner(micro_loss, cls_idx)

                    if not micro_loss_avg:
                        micro_loss_avg = micro_loss
                    else:
                        micro_loss_avg += micro_loss

                    micro_loss_count += 1
                micro_logit_dict[doc_id][cls_idx] = micro_logits

        if not train:  # if there are no predictions
            global_logit_dict = self.convert_from_micro_to_global_logits(micro_logit_dict)
            missing_preds = set(tmp_docids) - set(global_logit_dict.keys())
            if missing_preds:
                for p in missing_preds:
                    global_logit_dict[p] = torch.zeros(self.num_labels)
            outputs = (global_logit_dict,) + outputs[2:]  # add hidden states and attention if they are here

        if train:
            micro_loss = micro_loss_avg / micro_loss_count
            # micro_loss = micro_loss_avg

            # loss, cluster_loss, micro_loss = self.mtl_loss_combiner(cluster_loss, micro_loss)

            # in case all 8 examples in a batch are labelless, make sure we're returning a tensor
            micro_loss = torch.tensor(0.).cuda() if type(micro_loss) == float and micro_loss == 0.0 else micro_loss
            # micro_loss = torch.Tensor([micro_loss])

            loss = (cluster_loss, micro_loss)
            outputs = loss + outputs
        self.iteration += 1
        return outputs  # (cluster_loss, micro_loss), logits, (hidden_states), (attentions)


class MC2C_noMLCC(BertPreTrainedModel):
    def __init__(self, config, args='', loss_fct='', class_weights=None, doc_ids2_clusters=None, clusters=None,
                 cluster_idx2seed=None, cluster_output_dir=None):
        super().__init__(config)
        self.args = args
        self.config = config
        if self.args.encoder_type == 'bert':
            self.bert = BertModel(config)
        elif self.args.encoder_type == 'xlmroberta':
            self.roberta = XLMRobertaModel(config)
        self.num_labels = args.num_labels
        if not class_weights:
            self.class_weights = torch.ones((self.num_labels,))
        else:
            self.class_weights = class_weights
        self.iteration = 1
        self.loss_fct = loss_fct
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.n_clusters = len(clusters)
        self.clusters = clusters
        self.doc_ids2_clusters = doc_ids2_clusters
        self.cluster_idx2seed = cluster_idx2seed
        self.cluster_output_dir = cluster_output_dir
        self.create_MCC_classifiers()
        self.init_weights()
        self.cluster_activator_f1 = []
        self.cluster_activator_p = []
        self.cluster_activator_r = []

    def create_MCC_classifiers(self):
        """
        Dynamically generates appropriate number of inner-cluster classifiers, of the right size according to the
        number of labels in a given cluster. Allows for smooth experimenting with different numbers of clusters/
        different cluster sizes
        :return:
        """
        classifiers = {"self.mcc{}".format(str(i)): torch.nn.Linear(self.config.hidden_size,
                                                                    len(self.clusters[self.cluster_idx2seed[i]]))
                       for i in range(self.n_clusters)}
        for k, v in classifiers.items():
            exec("%s=%s" % (k, v))

    def get_idx2id(self, idx2id):
        self.idx2id = idx2id

    def load_local_labels(self, seed):
        return pickle.load(open(os.path.join(self.cluster_output_dir,'{}/train_doc_id2gold.p'.format(seed)), 'rb'))

    def load_local2global_idx(self, seed):
        return pickle.load(open(os.path.join(self.cluster_output_dir,
                                             '{}/cluster_idx2overall_idx.p'.format(seed)), 'rb'))

    def load_class_counts(self, local_or_global):
        if local_or_global == 'local':
            return pickle.load(open(os.path.join(self.cluster_output_dir, 'local_class_counts.p'), 'rb'))
        elif local_or_global == 'global':
            return pickle.load(open(os.path.join(self.cluster_output_dir, 'global_cluster_counts.p'), 'rb'))

    def convert_from_micro_to_global_logits(self, micro_logit_dict):
        global_logits = dict()
        for doc_id in micro_logit_dict.keys():
            pos_indices = []
            for cls_idx, logits in micro_logit_dict[doc_id].items():
                seed = self.cluster_idx2seed[cls_idx]
                local2global_dict = self.load_local2global_idx(seed)
                logits = nn.Softmax(dim=-1)(logits)
                label = torch.argmax(logits)
                pos_indices.append(local2global_dict[label.item()])

            logits = torch.zeros(self.num_labels)
            logits[pos_indices] = 1
            global_logits[doc_id] = logits
        return global_logits

    def freeze_bert_encoder(self):
        if self.args.n_bert2freeze:
            for name, param in self.bert.named_parameters():
                if name.split('.')[1] == 'encoder':
                    layer_n = name.split('.')[3]
                    if int(layer_n) < self.args.n_bert2freeze:
                        param.requires_grad = False
        else:
            for param in self.bert.parameters():
                param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
            self,
            doc_input_ids=None,
            doc_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            activated_clusters=None,
            labels=None,
            cluster_labels=None,
            local_labels=None,
            ranks=None,
            output_attentions=None,
            epoch=1,
            doc_ids=None,
            train=False,
            debug=False,
    ):
        if self.args.encoder_type == 'xlmroberta':
            outputs = self.roberta(
                doc_input_ids,
                attention_mask=doc_attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                # output_attentions=output_attentions,
            )
        elif self.args.encoder_type == 'bert':
            outputs = self.bert(
                doc_input_ids,
                attention_mask=doc_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                # output_attentions=output_attentions,
            )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        batch_size = pooled_output.shape[0]

        ###############################################################################################################
        #                         CLASSIFIERS TO PREDICT LABELS WITHIN THE CLUSTERS                                   #
        ###############################################################################################################

        cluster_labels = torch.ones(batch_size, self.n_clusters)

        micro_logit_dict = defaultdict(dict)
        # now we have the indices which tell us which classifiers we need to use
        # the row index --> example, columun index --> relevant classifier for that

        local_counts = self.load_class_counts('local')

        micro_loss_avg = 0
        micro_loss_count = 1
        tmp_docids = []
        cluster_labels2 = cluster_labels.clone().detach()
        for ex_idx, (ex_labels, doc_id) in enumerate(zip(cluster_labels2, doc_ids)):
            doc_id = self.idx2id[doc_id.item()]
            tmp_docids.append(doc_id)
            # if self.args.train_with_none and train:  # activates all classifiers
            #     ex_labels += 1
            relevant_classifiers = torch.nonzero(ex_labels)
            for cls_idx in relevant_classifiers:
                cls_idx = cls_idx.item()
                counts = torch.Tensor(
                    local_counts[cls_idx]).cuda()  # gives us the counts for the classes in that cluster
                # classifier = locals()['self.mcc{}'.format(cls_idx)].cuda() # call the relevant classifier
                classifier = getattr(self, 'mcc{}'.format(cls_idx)).cuda()  # call the relevant classifier
                micro_labels = self.load_local_labels(self.cluster_idx2seed[cls_idx])
                micro_logits = classifier(pooled_output[ex_idx, :])
                if train:
                    micro_labels = torch.Tensor(micro_labels[doc_id]).cuda()
                    if micro_logits.shape[-1] == 1:
                        micro_loss_fct = BCEWithLogitsLoss(
                            pos_weight=torch.Tensor([(self.args.n_examples - c) / c for c in counts]).cuda())
                        micro_loss = micro_loss_fct(micro_logits.view(-1, len(counts)),
                                                    micro_labels.view(-1, len(counts)))
                    elif self.args.mcc_loss == 'ldam':
                        micro_loss_fct = LDAMLoss(counts, max_m=self.args.max_m)
                        micro_loss = micro_loss_fct(micro_logits.view(-1, len(counts)),
                                                    micro_labels.view(-1, len(counts)))
                    elif self.args.mcc_loss == 'ce':
                        counts = torch.Tensor([(self.args.n_examples - c) / c for c in counts]).cuda()
                        if self.args.use_mcc_class_weights:
                            micro_loss_fct = CrossEntropyLoss(weight=counts)
                        else:
                            micro_loss_fct = CrossEntropyLoss()
                        micro_loss = micro_loss_fct(micro_logits.view(-1, len(counts)),
                                                    torch.nonzero(micro_labels, as_tuple=True)[0].cuda())

                    if not micro_loss_avg:
                        micro_loss_avg = micro_loss
                    else:
                        micro_loss_avg += micro_loss
                    micro_loss_count += 1
                micro_logit_dict[doc_id][cls_idx] = micro_logits

        if not train:
            global_logit_dict = self.convert_from_micro_to_global_logits(micro_logit_dict)
            missing_preds = set(tmp_docids) - set(global_logit_dict.keys())
            if missing_preds:
                for p in missing_preds:
                    global_logit_dict[p] = torch.zeros(self.num_labels)
            outputs = (global_logit_dict,) + outputs[2:]  # add hidden states and attention if they are here

        if train:
            micro_loss = micro_loss_avg / micro_loss_count
            outputs = (micro_loss,) + outputs
        self.iteration += 1
        return outputs  # (cluster_loss, micro_loss), logits, (hidden_states), (attentions)


class LabelAttentionMC2C(BertForMLSCWithLabelAttention):
    def __init__(self, config, args='', loss_fct='', class_weights=None, doc_ids2_clusters=None, clusters=None,
                 cluster_idx2seed=None, cluster_output_dir=None):
        BertForMLSCWithLabelAttention.__init__(self, config, args=args, loss_fct=loss_fct, class_weights=class_weights)
        self.config = config
        if args.train_with_none:
            self.num_labels -= 1
        self.label_attn_output_size = self.num_labels

        ###### DELETE #######
        # self.label_attn_output_size = self.args.doc_max_seq_length
        # self.w1 = torch.nn.Linear(self.num_labels * self.args.label_max_seq_length, 1)
        #####################

        self.n_clusters = len(clusters)
        self.clusters = clusters
        self.MLCC = torch.nn.Linear(self.label_attn_output_size, self.n_clusters)
        self.doc_ids2_clusters = doc_ids2_clusters
        self.cluster_idx2seed = cluster_idx2seed
        self.cluster_output_dir = cluster_output_dir
        self.create_MCC_classifiers()
        self.init_weights()
        self.cluster_activator_f1 = []
        self.cluster_activator_p = []
        self.cluster_activator_r = []
        self.step = 0

    def initialize_label_data(self, label_data):
        self.label_data = label_data

    def create_MCC_classifiers(self):
        """
        Dynamically generates appropriate number of inner-cluster classifiers, of the right size according to the
        number of labels in a given cluster. Allows for smooth experimenting with different numbers of clusters/
        different cluster sizes
        :return:
        """
        in_size = self.label_attn_output_size if not self.args.pass_mlcc_preds_to_mccs else \
            self.label_attn_output_size + self.n_clusters
        classifiers = {"self.mcc{}".format(str(i)): torch.nn.Linear(in_size,
                                                                    len(self.clusters[self.cluster_idx2seed[i]]))
                       for i in range(self.n_clusters)}
        for k, v in classifiers.items():
            exec("%s=%s" % (k, v))

    def get_idx2id(self, idx2id):
        self.idx2id = idx2id

    def load_local_labels(self, seed):
        return pickle.load(open(os.path.join(self.cluster_output_dir, '{}/train_doc_id2gold.p'.format(seed)), 'rb'))

    def load_local2global_idx(self, seed):
        return pickle.load(open(os.path.join(self.cluster_output_dir,
                                             '{}/cluster_idx2overall_idx.p'.format(seed)), 'rb'))

    def load_class_counts(self, local_or_global):
        if local_or_global == 'local':
            return pickle.load(open(os.path.join(self.cluster_output_dir, 'local_class_counts.p'), 'rb'))
        elif local_or_global == 'global':
            return pickle.load(open(os.path.join(self.cluster_output_dir, 'global_cluster_counts.p'), 'rb'))

    def convert_from_micro_to_global_logits(self, micro_logit_dict):
        global_logits = dict()
        for doc_id in micro_logit_dict.keys():
            pos_indices = []
            for cls_idx, logits in micro_logit_dict[doc_id].items():
                seed = self.cluster_idx2seed[cls_idx]
                local2global_dict = self.load_local2global_idx(seed)
                logits = nn.Softmax(dim=-1)(logits)
                label = torch.argmax(logits)
                pos_indices.append(local2global_dict[label.item()])

            logits = torch.zeros(self.num_labels + 1) if self.args.train_with_none else torch.zeros(self.num_labels)
            logits[pos_indices] = 1
            global_logits[doc_id] = logits
        return global_logits

    def freeze_bert_encoder(self):
        if self.args.n_bert2freeze:
            for name, param in self.bert.named_parameters():
                if name.split('.')[1] == 'encoder':
                    layer_n = name.split('.')[3]
                    if int(layer_n) < self.args.n_bert2freeze:
                        param.requires_grad = False
        else:
            for param in self.bert.parameters():
                param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
            self,
            doc_input_ids=None,
            doc_attention_mask=None,
            label_desc_input_ids=None,
            label_desc_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            ranks=None,
            output_attentions=None,
            epoch=1,
            doc_ids=None,
            train=False,
            activated_clusters=None,
            cluster_labels=None,
            local_labels=None,
            t_total=0,
            debug=False,
    ):
        self.step += 1

        if self.args.encoder_type == 'xlmroberta':
            doc_outputs = self.roberta(
                doc_input_ids,
                attention_mask=doc_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            label_outputs = self.roberta(
                self.label_data[0].cuda(),
                attention_mask=self.label_data[1].cuda(),
            )
        elif self.args.encoder_type == 'bert':
            doc_outputs = self.bert(
                doc_input_ids,
                attention_mask=doc_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                # output_attentions=output_attentions,
            )

            label_outputs = self.bert(
                self.label_data[0].cuda(),
                attention_mask=self.label_data[1].cuda(),
                token_type_ids=self.label_data[-1].cuda(),
            )

        # get the sequence-level document representations
        doc_seq_output = doc_outputs[0]
        doc_seq_output - self.dropout(doc_seq_output)

        batch_size = doc_seq_output.shape[0]

        # get the sequence-level label description representations
        label_seq_output = label_outputs[0]

        label_seq_output = label_seq_output.reshape(self.num_labels * self.args.label_max_seq_length, self.hidden_size)
        temp = torch.matmul(doc_seq_output, label_seq_output.T)

        ###### UNCOMMENT #######
        temp = temp.permute(0, 2, 1)
        ########################

        # logits = temp.view(self.args.per_gpu_train_batch_size, self.label_attn_output_size)

        temp = self.w1(temp)

        ###### DELETE #######
        # logits = temp.view(-1, self.label_attn_output_size)
        #####################

        ###### UNCOMMENT #######
        temp = temp.reshape(batch_size, self.num_labels, self.args.label_max_seq_length)
        temp = self.w2(temp)
        logits = temp.view(-1, self.num_labels)
        ########################

        #
        # if self.args.doc_batching:
        #     if self.args.logit_aggregation == 'max':
        #         logits = torch.max(logits, axis=0)[0]
        #     elif self.args.logit_aggregation == 'avg':
        #         logits = torch.mean(logits, axis=0)

        ###############################################################################################################
        #                          FIRST CLASSIFIER TO PREDICT WHICH CLUSTERS ARE ACTIVATED                           #
        ###############################################################################################################
        # now we have our CLS vector
        cluster_logits = self.MLCC(logits)  # batch_size, n_clusters

        # print("CLUSTER LOGITS:", cluster_logits)
        batch_size = cluster_logits.shape[0]

        ids = [self.idx2id[i.item()] for i in doc_ids]
        if train or self.args.eval_cluster_activator:
            cluster_labels = torch.Tensor([self.doc_ids2_clusters[d] for d in ids]).cuda()
            cluster_weights = torch.Tensor(self.load_class_counts('global')).cuda()
            if self.args.loss_fct == 'bce':
                if self.args.use_bce_class_weights:
                    cluster_loss = BCEWithLogitsLoss(pos_weight=cluster_weights)(cluster_logits, cluster_labels)
                else:
                    cluster_loss = BCEWithLogitsLoss()(cluster_logits, cluster_labels)
            elif self.args.loss_fct == 'bbce':
                cluster_loss = BalancedBCEWithLogitsLoss(grad_clip=True)(cluster_logits, cluster_labels)
            if self.args.eval_cluster_activator:
                tmp = torch.nn.Sigmoid()(cluster_logits)
                tmp = (tmp > 0.5)
                f1 = f1_score(y_true=cluster_labels.cpu(), y_pred=tmp.cpu(), average=self.args.metric_avg)
                p = precision_score(y_true=cluster_labels.cpu(), y_pred=tmp.cpu(), average=self.args.metric_avg)
                r = recall_score(y_true=cluster_labels.cpu(), y_pred=tmp.cpu(), average=self.args.metric_avg)
                self.cluster_activator_f1.append(f1)
                self.cluster_activator_p.append(p)
                self.cluster_activator_r.append(r)

        ###############################################################################################################
        ###############################################################################################################

        ###############################################################################################################
        #                          SECOND CLASSIFIER TO PREDICT LABELS WITHIN THE CLUSTERS                            #
        ###############################################################################################################
        if not train:
            cluster_labels = torch.zeros(cluster_logits.shape)
            cluster_labels_idx = torch.nonzero((torch.nn.Sigmoid()(cluster_logits) > 0.5).int(), as_tuple=True)
            cluster_labels[cluster_labels_idx] = 1


        micro_logit_dict = defaultdict(dict)
        # now we have the indices which tell us which classifiers we need to use
        # the row index --> example, columun index --> relevant classifier for that

        local_counts = self.load_class_counts('local')

        micro_loss_avg = 0
        micro_loss_count = 1
        tmp_docids = []
        cluster_labels2 = cluster_labels.clone().detach()
        for ex_idx, (ex_labels, doc_id) in enumerate(zip(cluster_labels2, doc_ids)):
            doc_id = self.idx2id[doc_id.item()]
            tmp_docids.append(doc_id)
            if self.args.train_with_none and train:
                ex_labels += 1
            relevant_classifiers = torch.nonzero(ex_labels)
            for cls_idx in relevant_classifiers:
                cls_idx = cls_idx.item()
                counts = torch.Tensor(
                    local_counts[cls_idx]).cuda()  # gives us the counts for the classes in that cluster
                # classifier = locals()['self.mcc{}'.format(cls_idx)].cuda() # call the relevant classifier
                classifier = getattr(self, 'mcc{}'.format(cls_idx)).cuda()  # call the relevant classifier
                # classifier = self.MCClassifiers[cls_idx].cuda()  # find the relevant classifier
                micro_labels = self.load_local_labels(self.cluster_idx2seed[cls_idx])
                if self.args.pass_mlcc_preds_to_mccs:
                    if self.step <= t_total * .1:
                        mcc_input = torch.cat((logits[ex_idx, :], torch.nn.Sigmoid()(cluster_labels2[ex_idx, :].cuda())))
                    else:
                        mcc_input = torch.cat((logits[ex_idx, :], torch.nn.Sigmoid()(cluster_logits[ex_idx, :].cuda())))
                else:
                    mcc_input = logits[ex_idx, :]
                micro_logits = classifier(mcc_input)
                if train:
                    micro_labels = torch.Tensor(micro_labels[doc_id]).cuda()
                    if micro_logits.shape[-1] == 1:
                        micro_loss_fct = BCEWithLogitsLoss(
                            pos_weight=torch.Tensor([(self.args.n_examples - c) / c for c in counts]).cuda())
                        micro_loss = micro_loss_fct(micro_logits.view(-1, len(counts)),
                                                    micro_labels.view(-1, len(counts)))
                    elif self.args.mcc_loss == 'ldam':
                        micro_loss_fct = LDAMLoss(counts, max_m=self.args.max_m)
                        micro_loss = micro_loss_fct(micro_logits.view(-1, len(counts)),
                                                    micro_labels.view(-1, len(counts)))
                    elif self.args.mcc_loss == 'ce':
                        counts = torch.Tensor([(self.args.n_examples - c) / c for c in counts]).cuda()
                        if self.args.use_mcc_class_weights:
                            micro_loss_fct = CrossEntropyLoss(weight=counts)
                        else:
                            micro_loss_fct = CrossEntropyLoss()
                        # micro_loss_fct = CrossEntropyLoss()
                        micro_loss = micro_loss_fct(micro_logits.view(-1, len(counts)),
                                                    torch.nonzero(micro_labels, as_tuple=True)[0].cuda())

                    if self.args.mlcc_as_gates:
                        # we can use the MLCC predictions as weights for the different MCC losses (gates)
                        gate = torch.nn.Sigmoid()(cluster_logits[ex_idx, cls_idx])
                        micro_loss *= gate

                    if not micro_loss_avg:
                        micro_loss_avg = micro_loss
                    else:
                        micro_loss_avg += micro_loss
                    micro_loss_count += 1
                micro_logit_dict[doc_id][cls_idx] = micro_logits

        if not train:
            global_logit_dict = self.convert_from_micro_to_global_logits(micro_logit_dict)
            missing_preds = set(tmp_docids) - set(global_logit_dict.keys())
            if missing_preds:
                for p in missing_preds:
                    n = self.num_labels if not self.args.train_with_none else self.num_labels + 1
                    global_logit_dict[p] = torch.zeros(n)
            outputs = (global_logit_dict,) + doc_outputs[2:]  # add hidden states and attention if they are here

        if train:
            micro_loss = micro_loss_avg / micro_loss_count
            # loss, cluster_loss, micro_loss = self.mtl_loss_combiner(cluster_loss, micro_loss)

            # in case all 8 examples in a batch are labelless, make sure we're returning a tensor
            micro_loss = torch.tensor(0.).cuda() if type(micro_loss) == float and micro_loss == 0.0 else micro_loss
            outputs = (cluster_loss, micro_loss) + doc_outputs
            # loss = (cluster_loss, micro_loss)
            # outputs = loss + outputs
        self.iteration += 1
        return outputs  # (cluster_loss, micro_loss), logits, (hidden_states), (attentions)


class LabelAttentionMC2C_Experimental(BertForMLSCWithLabelAttention):
    def __init__(self, config, args='', loss_fct='', class_weights=None, doc_ids2_clusters=None, clusters=None,
                 cluster_idx2seed=None, cluster_output_dir=None):
        BertForMLSCWithLabelAttention.__init__(self, config, args=args, loss_fct=loss_fct, class_weights=class_weights)
        self.config = config
        self.label_attn_output_size = self.num_labels

        ###### DELETE #######
        # self.label_attn_output_size = self.args.doc_max_seq_length
        # self.w1 = torch.nn.Linear(self.num_labels, 1)
        self.w1 = torch.nn.Linear(self.args.label_max_seq_length, 1)
        self.w2 = torch.nn.Linear(self.args.doc_max_seq_length, 1)
        #####################

        self.n_clusters = len(clusters)
        self.clusters = clusters
        self.MLCC = torch.nn.Linear(self.label_attn_output_size, self.n_clusters)
        self.doc_ids2_clusters = doc_ids2_clusters
        self.cluster_idx2seed = cluster_idx2seed
        self.cluster_output_dir = cluster_output_dir
        self.create_MCC_classifiers()
        self.init_weights()
        self.cluster_activator_f1 = []
        self.cluster_activator_p = []
        self.cluster_activator_r = []
        self.step = 0

    def initialize_label_data(self, label_data):
        self.label_data = label_data

    def create_MCC_classifiers(self):
        """
        Dynamically generates appropriate number of inner-cluster classifiers, of the right size according to the
        number of labels in a given cluster. Allows for smooth experimenting with different numbers of clusters/
        different cluster sizes
        :return:
        """
        in_size = self.label_attn_output_size if not self.args.pass_mlcc_preds_to_mccs else \
            self.label_attn_output_size + self.n_clusters
        classifiers = {"self.mcc{}".format(str(i)): torch.nn.Linear(in_size,
                                                                    len(self.clusters[self.cluster_idx2seed[i]]))
                       for i in range(self.n_clusters)}
        for k, v in classifiers.items():
            exec("%s=%s" % (k, v))

    def get_idx2id(self, idx2id):
        self.idx2id = idx2id

    def load_local_labels(self, seed):
        return pickle.load(open(os.path.join(self.cluster_output_dir,'{}/train_doc_id2gold.p'.format(seed)), 'rb'))

    def load_local2global_idx(self, seed):
        return pickle.load(open(os.path.join(self.cluster_output_dir,
                                             '{}/cluster_idx2overall_idx.p'.format(seed)), 'rb'))

    def load_class_counts(self, local_or_global):
        if local_or_global == 'local':
            return pickle.load(open(os.path.join(self.cluster_output_dir, 'local_class_counts.p'), 'rb'))
        elif local_or_global == 'global':
            return pickle.load(open(os.path.join(self.cluster_output_dir, 'global_cluster_counts.p'), 'rb'))

    def convert_from_micro_to_global_logits(self, micro_logit_dict):
        global_logits = dict()
        for doc_id in micro_logit_dict.keys():
            pos_indices = []
            for cls_idx, logits in micro_logit_dict[doc_id].items():
                seed = self.cluster_idx2seed[cls_idx]
                local2global_dict = self.load_local2global_idx(seed)
                logits = nn.Softmax(dim=-1)(logits)
                label = torch.argmax(logits)
                pos_indices.append(local2global_dict[label.item()])

            logits = torch.zeros(self.num_labels)
            logits[pos_indices] = 1
            global_logits[doc_id] = logits
        return global_logits

    def freeze_bert_encoder(self):
        if self.args.n_bert2freeze:
            for name, param in self.bert.named_parameters():
                if name.split('.')[1] == 'encoder':
                    layer_n = name.split('.')[3]
                    if int(layer_n) < self.args.n_bert2freeze:
                        param.requires_grad = False
        else:
            for param in self.bert.parameters():
                param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
            self,
            doc_input_ids=None,
            doc_attention_mask=None,
            label_desc_input_ids=None,
            label_desc_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            ranks=None,
            output_attentions=None,
            epoch=1,
            doc_ids=None,
            train=False,
            activated_clusters=None,
            cluster_labels=None,
            local_labels=None,
            t_total=0,
            debug=False,
    ):
        self.step += 1

        if self.args.encoder_type == 'xlmroberta':
            doc_outputs = self.roberta(
                doc_input_ids,
                attention_mask=doc_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            label_outputs = self.roberta(
                self.label_data[0].cuda(),
                attention_mask=self.label_data[1].cuda(),
            )
        elif self.args.encoder_type == 'bert':
            doc_outputs = self.bert(
                doc_input_ids,
                attention_mask=doc_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                # output_attentions=output_attentions,
            )

            label_outputs = self.bert(
                self.label_data[0].cuda(),
                attention_mask=self.label_data[1].cuda(),
                token_type_ids=self.label_data[-1].cuda(),
            )

        # get the sequence-level document representations
        doc_seq_output = doc_outputs[0]
        doc_seq_output - self.dropout(doc_seq_output)

        batch_size = doc_seq_output.shape[0]

        # get the sequence-level label description representations
        label_seq_output = label_outputs[0]

        label_seq_output = label_seq_output.reshape(self.num_labels * self.args.label_max_seq_length, self.hidden_size)



        attn_scores = torch.matmul(doc_seq_output, label_seq_output.T)  # batch_size, doc msl, label msl * #labels
                                                                 # a matrix (1 per doc) of word-level attention scores
                                                                 # between input docs & words in label desc


        attn_scores = attn_scores.reshape(batch_size,
                                          self.args.doc_max_seq_length,
                                          self.num_labels,
                                          self.args.label_max_seq_length)


        attn_scores = self.w1(attn_scores)
        attn_scores = attn_scores.view(-1, self.num_labels)
        attn_scores = attn_scores.reshape(batch_size, self.args.doc_max_seq_length, self.num_labels)



        attn_scores = attn_scores.permute(0,2,1)

        logits = self.w2(attn_scores)
        logits = logits.view(-1, self.label_attn_output_size)  # batch_size, doc msl OR # labels


        # ###### UNCOMMENT #######
        # temp = torch.matmul(doc_seq_output, label_seq_output.T)
        # temp = temp.permute(0, 2, 1)
        # ########################
        #
        # # logits = temp.view(self.args.per_gpu_train_batch_size, self.label_attn_output_size)
        #
        # temp = self.w1(temp)
        #
        # ###### DELETE #######
        # # logits = temp.view(-1, self.label_attn_output_size)
        # #####################
        #
        # ###### UNCOMMENT #######
        # temp = temp.reshape(batch_size, self.num_labels, self.args.label_max_seq_length)
        # temp = self.w2(temp)
        # logits = temp.view(-1, self.num_labels)
        # ########################

        #
        # if self.args.doc_batching:
        #     if self.args.logit_aggregation == 'max':
        #         logits = torch.max(logits, axis=0)[0]
        #     elif self.args.logit_aggregation == 'avg':
        #         logits = torch.mean(logits, axis=0)

        ###############################################################################################################
        #                          FIRST CLASSIFIER TO PREDICT WHICH CLUSTERS ARE ACTIVATED                           #
        ###############################################################################################################
        # now we have our CLS vector
        cluster_logits = self.MLCC(logits)  # batch_size, n_clusters

        # print("CLUSTER LOGITS:", cluster_logits)
        batch_size = cluster_logits.shape[0]

        ids = [self.idx2id[i.item()] for i in doc_ids]
        if train or self.args.eval_cluster_activator:
            cluster_labels = torch.Tensor([self.doc_ids2_clusters[d] for d in ids]).cuda()
            cluster_weights = torch.Tensor(self.load_class_counts('global')).cuda()
            if self.args.loss_fct == 'bce':
                if self.args.use_bce_class_weights:
                    cluster_loss = BCEWithLogitsLoss(pos_weight=cluster_weights)(cluster_logits, cluster_labels)
                else:
                    cluster_loss = BCEWithLogitsLoss()(cluster_logits, cluster_labels)
            elif self.args.loss_fct == 'bbce':
                cluster_loss = BalancedBCEWithLogitsLoss(grad_clip=True)(cluster_logits, cluster_labels)
            if self.args.eval_cluster_activator:
                tmp = torch.nn.Sigmoid()(cluster_logits)
                tmp = (tmp > 0.5)
                f1 = f1_score(y_true=cluster_labels.cpu(), y_pred=tmp.cpu(), average=self.args.metric_avg)
                p = precision_score(y_true=cluster_labels.cpu(), y_pred=tmp.cpu(), average=self.args.metric_avg)
                r = recall_score(y_true=cluster_labels.cpu(), y_pred=tmp.cpu(), average=self.args.metric_avg)
                self.cluster_activator_f1.append(f1)
                self.cluster_activator_p.append(p)
                self.cluster_activator_r.append(r)

        ###############################################################################################################
        ###############################################################################################################

        ###############################################################################################################
        #                          SECOND CLASSIFIER TO PREDICT LABELS WITHIN THE CLUSTERS                            #
        ###############################################################################################################
        if not train:
            cluster_labels = torch.zeros(cluster_logits.shape)
            cluster_labels_idx = torch.nonzero((torch.nn.Sigmoid()(cluster_logits) > 0.5).int(), as_tuple=True)
            cluster_labels[cluster_labels_idx] = 1


        micro_logit_dict = defaultdict(dict)
        # now we have the indices which tell us which classifiers we need to use
        # the row index --> example, columun index --> relevant classifier for that

        local_counts = self.load_class_counts('local')

        micro_loss_avg = 0
        micro_loss_count = 1
        tmp_docids = []
        cluster_labels2 = cluster_labels.clone().detach()
        for ex_idx, (ex_labels, doc_id) in enumerate(zip(cluster_labels2, doc_ids)):
            doc_id = self.idx2id[doc_id.item()]
            tmp_docids.append(doc_id)
            if self.args.train_with_none and train:
                ex_labels += 1
            relevant_classifiers = torch.nonzero(ex_labels)
            for cls_idx in relevant_classifiers:
                cls_idx = cls_idx.item()
                counts = torch.Tensor(
                    local_counts[cls_idx]).cuda()  # gives us the counts for the classes in that cluster
                # classifier = locals()['self.mcc{}'.format(cls_idx)].cuda() # call the relevant classifier
                classifier = getattr(self, 'mcc{}'.format(cls_idx)).cuda()  # call the relevant classifier
                # classifier = self.MCClassifiers[cls_idx].cuda()  # find the relevant classifier
                micro_labels = self.load_local_labels(self.cluster_idx2seed[cls_idx])
                if self.args.pass_mlcc_preds_to_mccs:
                    if self.step <= t_total * .1:
                        mcc_input = torch.cat((logits[ex_idx, :], torch.nn.Sigmoid()(cluster_labels2[ex_idx, :].cuda())))
                    else:
                        mcc_input = torch.cat((logits[ex_idx, :], torch.nn.Sigmoid()(cluster_logits[ex_idx, :].cuda())))
                else:
                    mcc_input = logits[ex_idx, :]
                micro_logits = classifier(mcc_input)
                if train:
                    micro_labels = torch.Tensor(micro_labels[doc_id]).cuda()
                    if micro_logits.shape[-1] == 1:
                        micro_loss_fct = BCEWithLogitsLoss(
                            pos_weight=torch.Tensor([(self.args.n_examples - c) / c for c in counts]).cuda())
                        micro_loss = micro_loss_fct(micro_logits.view(-1, len(counts)),
                                                    micro_labels.view(-1, len(counts)))
                    elif self.args.mcc_loss == 'ldam':
                        micro_loss_fct = LDAMLoss(counts, max_m=self.args.max_m)
                        micro_loss = micro_loss_fct(micro_logits.view(-1, len(counts)),
                                                    micro_labels.view(-1, len(counts)))
                    elif self.args.mcc_loss == 'ce':
                        counts = torch.Tensor([(self.args.n_examples - c) / c for c in counts]).cuda()
                        if self.args.use_mcc_class_weights:
                            micro_loss_fct = CrossEntropyLoss(weight=counts)
                        else:
                            micro_loss_fct = CrossEntropyLoss()
                        # micro_loss_fct = CrossEntropyLoss()
                        micro_loss = micro_loss_fct(micro_logits.view(-1, len(counts)),
                                                    torch.nonzero(micro_labels, as_tuple=True)[0].cuda())

                    if self.args.mlcc_as_gates:
                        # we can use the MLCC predictions as weights for the different MCC losses (gates)
                        gate = torch.nn.Sigmoid()(cluster_logits[ex_idx, cls_idx])
                        micro_loss *= gate


                    if not micro_loss_avg:
                        micro_loss_avg = micro_loss
                    else:
                        micro_loss_avg += micro_loss
                    micro_loss_count += 1
                micro_logit_dict[doc_id][cls_idx] = micro_logits

        if not train:
            global_logit_dict = self.convert_from_micro_to_global_logits(micro_logit_dict)
            missing_preds = set(tmp_docids) - set(global_logit_dict.keys())
            if missing_preds:
                for p in missing_preds:
                    global_logit_dict[p] = torch.zeros(self.num_labels)
            outputs = (global_logit_dict,) + doc_outputs[2:]  # add hidden states and attention if they are here

        if train:
            micro_loss = micro_loss_avg / micro_loss_count
            # loss, cluster_loss, micro_loss = self.mtl_loss_combiner(cluster_loss, micro_loss)

            # in case all 8 examples in a batch are labelless, make sure we're returning a tensor
            micro_loss = torch.tensor(0.).cuda() if type(micro_loss) == float and micro_loss == 0.0 else micro_loss
            outputs = (cluster_loss, micro_loss) + doc_outputs
            # loss = (cluster_loss, micro_loss)
            # outputs = loss + outputs
        self.iteration += 1
        return outputs  # (cluster_loss, micro_loss), logits, (hidden_states), (attentions)


class LabelAttentionMC2C_Exp2(BertForMLSCWithLabelAttention):
    def __init__(self, config, args='', loss_fct='', class_weights=None, doc_ids2_clusters=None, clusters=None,
                 cluster_idx2seed=None, cluster_output_dir=None):
        BertForMLSCWithLabelAttention.__init__(self, config, args=args, loss_fct=loss_fct, class_weights=class_weights)
        self.config = config
        if args.train_with_none:
            self.num_labels -= 1
        self.label_attn_output_size = self.num_labels

        ###### DELETE #######
        # self.label_attn_output_size = self.args.doc_max_seq_length
        # self.w1 = torch.nn.Linear(self.num_labels * self.args.label_max_seq_length, 1)
        #####################

        self.n_clusters = len(clusters)
        self.clusters = clusters
        self.MLCC = torch.nn.Linear(self.label_attn_output_size, self.n_clusters)
        self.doc_ids2_clusters = doc_ids2_clusters
        self.cluster_idx2seed = cluster_idx2seed
        self.cluster_output_dir = cluster_output_dir
        self.create_MCC_classifiers()
        self.init_weights()
        self.cluster_activator_f1 = []
        self.cluster_activator_p = []
        self.cluster_activator_r = []
        self.step = 0

    def initialize_label_data(self, label_data):
        self.label_data = label_data

    def create_MCC_classifiers(self):
        """
        Dynamically generates appropriate number of inner-cluster classifiers, of the right size according to the
        number of labels in a given cluster. Allows for smooth experimenting with different numbers of clusters/
        different cluster sizes
        :return:
        """
        in_size = self.label_attn_output_size if not self.args.pass_mlcc_preds_to_mccs else \
            self.label_attn_output_size + self.n_clusters


        classifiers = {"self.mcc{}".format(str(i)): torch.nn.Linear(self.hidden_size+len(self.clusters[self.cluster_idx2seed[i]]),
                                                                    len(self.clusters[self.cluster_idx2seed[i]]))
                       for i in range(self.n_clusters)}
        for k, v in classifiers.items():
            exec("%s=%s" % (k, v))

    def get_idx2id(self, idx2id):
        self.idx2id = idx2id

    def load_local_labels(self, seed):
        return pickle.load(open(os.path.join(self.cluster_output_dir, '{}/train_doc_id2gold.p'.format(seed)), 'rb'))

    def load_local2global_idx(self, seed):
        return pickle.load(open(os.path.join(self.cluster_output_dir,
                                             '{}/cluster_idx2overall_idx.p'.format(seed)), 'rb'))

    def load_class_counts(self, local_or_global):
        if local_or_global == 'local':
            return pickle.load(open(os.path.join(self.cluster_output_dir, 'local_class_counts.p'), 'rb'))
        elif local_or_global == 'global':
            return pickle.load(open(os.path.join(self.cluster_output_dir, 'global_cluster_counts.p'), 'rb'))

    def convert_from_micro_to_global_logits(self, micro_logit_dict):
        global_logits = dict()
        for doc_id in micro_logit_dict.keys():
            pos_indices = []
            for cls_idx, logits in micro_logit_dict[doc_id].items():
                seed = self.cluster_idx2seed[cls_idx]
                local2global_dict = self.load_local2global_idx(seed)
                logits = nn.Softmax(dim=-1)(logits)
                label = torch.argmax(logits)
                pos_indices.append(local2global_dict[label.item()])

            logits = torch.zeros(self.num_labels + 1) if self.args.train_with_none else torch.zeros(self.num_labels)
            logits[pos_indices] = 1
            global_logits[doc_id] = logits
        return global_logits

    def freeze_bert_encoder(self):
        if self.args.n_bert2freeze:
            for name, param in self.bert.named_parameters():
                if name.split('.')[1] == 'encoder':
                    layer_n = name.split('.')[3]
                    if int(layer_n) < self.args.n_bert2freeze:
                        param.requires_grad = False
        else:
            for param in self.bert.parameters():
                param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def label_desc_attn(self, doc_reps, label_reps, mcc=False):
        batch_size = doc_reps.shape[0]
        n_labels = label_reps.shape[0]

        label_reps = label_reps.reshape(n_labels * self.args.label_max_seq_length, self.hidden_size)

        temp = torch.matmul(doc_reps, label_reps.T)

        ###### UNCOMMENT #######
        temp = temp.T if mcc else temp.permute(0, 2, 1)
        ########################

        # logits = temp.view(self.args.per_gpu_train_batch_size, self.label_attn_output_size)
        # if not mcc:
        #     w1, w2 = self.w1, self.w2
        # else:
        #     w1, w2 = self.w3, self.w4

        w1, w2 = self.w1, self.w2

        temp = w1(temp)

        ###### DELETE #######
        # logits = temp.view(-1, self.label_attn_output_size)
        #####################

        ###### UNCOMMENT #######
        temp = temp.reshape(n_labels, self.args.label_max_seq_length) if mcc else temp.reshape(batch_size, n_labels, self.args.label_max_seq_length)
        temp = w2(temp)
        logits = temp.view(-1, n_labels)
        ########################
        return logits

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
            self,
            doc_input_ids=None,
            doc_attention_mask=None,
            label_desc_input_ids=None,
            label_desc_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            ranks=None,
            output_attentions=None,
            epoch=1,
            doc_ids=None,
            train=False,
            activated_clusters=None,
            cluster_labels=None,
            local_labels=None,
            t_total=0,
            debug=False,
    ):
        self.step += 1

        if self.args.encoder_type == 'xlmroberta':
            doc_outputs = self.roberta(
                doc_input_ids,
                attention_mask=doc_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            label_outputs = self.roberta(
                self.label_data[0].cuda(),
                attention_mask=self.label_data[1].cuda(),
            )
        elif self.args.encoder_type == 'bert':
            doc_outputs = self.bert(
                doc_input_ids,
                attention_mask=doc_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                # output_attentions=output_attentions,
            )

            label_outputs = self.bert(
                self.label_data[0].cuda(),
                attention_mask=self.label_data[1].cuda(),
                token_type_ids=self.label_data[-1].cuda(),
            )

        # get the sequence-level document representations
        doc_seq_output = doc_outputs[0]
        doc_seq_output - self.dropout(doc_seq_output)


        # get the sequence-level label description representations
        label_seq_output = label_outputs[0]


        logits = self.label_desc_attn(doc_seq_output, label_seq_output)

        #
        # if self.args.doc_batching:
        #     if self.args.logit_aggregation == 'max':
        #         logits = torch.max(logits, axis=0)[0]
        #     elif self.args.logit_aggregation == 'avg':
        #         logits = torch.mean(logits, axis=0)

        ###############################################################################################################
        #                          FIRST CLASSIFIER TO PREDICT WHICH CLUSTERS ARE ACTIVATED                           #
        ###############################################################################################################
        # now we have our CLS vector
        cluster_logits = self.MLCC(logits)  # batch_size, n_clusters

        # print("CLUSTER LOGITS:", cluster_logits)
        batch_size = cluster_logits.shape[0]

        ids = [self.idx2id[i.item()] for i in doc_ids]
        if train or self.args.eval_cluster_activator:
            cluster_labels = torch.Tensor([self.doc_ids2_clusters[d] for d in ids]).cuda()
            cluster_weights = torch.Tensor(self.load_class_counts('global')).cuda()
            if self.args.loss_fct == 'bce':
                if self.args.use_bce_class_weights:
                    cluster_loss = BCEWithLogitsLoss(pos_weight=cluster_weights)(cluster_logits, cluster_labels)
                else:
                    cluster_loss = BCEWithLogitsLoss()(cluster_logits, cluster_labels)
            elif self.args.loss_fct == 'bbce':
                cluster_loss = BalancedBCEWithLogitsLoss(grad_clip=True)(cluster_logits, cluster_labels)
            if self.args.eval_cluster_activator:
                tmp = torch.nn.Sigmoid()(cluster_logits)
                tmp = (tmp > 0.5)
                f1 = f1_score(y_true=cluster_labels.cpu(), y_pred=tmp.cpu(), average=self.args.metric_avg)
                p = precision_score(y_true=cluster_labels.cpu(), y_pred=tmp.cpu(), average=self.args.metric_avg)
                r = recall_score(y_true=cluster_labels.cpu(), y_pred=tmp.cpu(), average=self.args.metric_avg)
                self.cluster_activator_f1.append(f1)
                self.cluster_activator_p.append(p)
                self.cluster_activator_r.append(r)

        ###############################################################################################################
        ###############################################################################################################

        ###############################################################################################################
        #                          SECOND CLASSIFIER TO PREDICT LABELS WITHIN THE CLUSTERS                            #
        ###############################################################################################################
        if not train:
            cluster_labels = torch.zeros(cluster_logits.shape)
            cluster_labels_idx = torch.nonzero((torch.nn.Sigmoid()(cluster_logits) > 0.5).int(), as_tuple=True)
            cluster_labels[cluster_labels_idx] = 1


        micro_logit_dict = defaultdict(dict)
        # now we have the indices which tell us which classifiers we need to use
        # the row index --> example, columun index --> relevant classifier for that

        local_counts = self.load_class_counts('local')

        micro_loss_avg = 0
        micro_loss_count = 1
        tmp_docids = []
        cluster_labels2 = cluster_labels.clone().detach()
        for ex_idx, (ex_labels, doc_id) in enumerate(zip(cluster_labels2, doc_ids)):
            doc_id = self.idx2id[doc_id.item()]
            tmp_docids.append(doc_id)
            if self.args.train_with_none and train:
                ex_labels += 1
            relevant_classifiers = torch.nonzero(ex_labels)
            for cls_idx in relevant_classifiers:
                cls_idx = cls_idx.item()
                counts = torch.Tensor(
                    local_counts[cls_idx]).cuda()  # gives us the counts for the classes in that cluster
                # classifier = locals()['self.mcc{}'.format(cls_idx)].cuda() # call the relevant classifier
                classifier = getattr(self, 'mcc{}'.format(cls_idx)).cuda()  # call the relevant classifier
                # classifier = self.MCClassifiers[cls_idx].cuda()  # find the relevant classifier
                micro_labels = self.load_local_labels(self.cluster_idx2seed[cls_idx])
                if self.args.pass_mlcc_preds_to_mccs:
                    if self.step <= t_total * .1:
                        mcc_input = torch.cat((logits[ex_idx, :], torch.nn.Sigmoid()(cluster_labels2[ex_idx, :].cuda())))
                    else:
                        mcc_input = torch.cat((logits[ex_idx, :], torch.nn.Sigmoid()(cluster_logits[ex_idx, :].cuda())))

                else:
                    mcc_input = logits[ex_idx, :]

                local2global_dict = self.load_local2global_idx(self.cluster_idx2seed[cls_idx])
                # mcc_input = label_seq_output[list(local2global_dict.values()),:,:]
                mcc_input = self.label_desc_attn(doc_seq_output[ex_idx, :], label_seq_output[list(local2global_dict.values()),:,:], mcc=True)

                # print(doc_outputs[1][ex_idx, :].shape)
                # print(mcc_input.shape)
                mcc_input = torch.cat((doc_outputs[1][ex_idx, :], mcc_input.view(-1)))

                micro_logits = classifier(mcc_input)


                if train:
                    micro_labels = torch.Tensor(micro_labels[doc_id]).cuda()
                    if micro_logits.shape[-1] == 1:
                        micro_loss_fct = BCEWithLogitsLoss(
                            pos_weight=torch.Tensor([(self.args.n_examples - c) / c for c in counts]).cuda())
                        micro_loss = micro_loss_fct(micro_logits.view(-1, len(counts)),
                                                    micro_labels.view(-1, len(counts)))
                    elif self.args.mcc_loss == 'ldam':
                        micro_loss_fct = LDAMLoss(counts, max_m=self.args.max_m)
                        micro_loss = micro_loss_fct(micro_logits.view(-1, len(counts)),
                                                    micro_labels.view(-1, len(counts)))
                    elif self.args.mcc_loss == 'ce':
                        counts = torch.Tensor([(self.args.n_examples - c) / c for c in counts]).cuda()
                        if self.args.use_mcc_class_weights:
                            micro_loss_fct = CrossEntropyLoss(weight=counts)
                        else:
                            micro_loss_fct = CrossEntropyLoss()
                        # micro_loss_fct = CrossEntropyLoss()
                        micro_loss = micro_loss_fct(micro_logits.view(-1, len(counts)),
                                                    torch.nonzero(micro_labels, as_tuple=True)[0].cuda())

                    if self.args.mlcc_as_gates:
                        # we can use the MLCC predictions as weights for the different MCC losses (gates)
                        gate = torch.nn.Sigmoid()(cluster_logits[ex_idx, cls_idx])
                        micro_loss *= gate

                    if not micro_loss_avg:
                        micro_loss_avg = micro_loss
                    else:
                        micro_loss_avg += micro_loss
                    micro_loss_count += 1
                micro_logit_dict[doc_id][cls_idx] = micro_logits

        if not train:
            global_logit_dict = self.convert_from_micro_to_global_logits(micro_logit_dict)
            missing_preds = set(tmp_docids) - set(global_logit_dict.keys())
            if missing_preds:
                for p in missing_preds:
                    n = self.num_labels if not self.args.train_with_none else self.num_labels + 1
                    global_logit_dict[p] = torch.zeros(n)
            outputs = (global_logit_dict,) + doc_outputs[2:]  # add hidden states and attention if they are here

        if train:
            micro_loss = micro_loss_avg / micro_loss_count
            # loss, cluster_loss, micro_loss = self.mtl_loss_combiner(cluster_loss, micro_loss)

            # in case all 8 examples in a batch are labelless, make sure we're returning a tensor
            micro_loss = torch.tensor(0.).cuda() if type(micro_loss) == float and micro_loss == 0.0 else micro_loss
            outputs = (cluster_loss, micro_loss) + doc_outputs
            # loss = (cluster_loss, micro_loss)
            # outputs = loss + outputs
        self.iteration += 1
        return outputs  # (cluster_loss, micro_loss), logits, (hidden_states), (attentions)
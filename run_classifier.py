import pandas as pd
import sys

# sys.path.insert(1, '../CLEF_Datasets_ICD/processed_data/')
from process_data import *
import torch
import io
import re
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
from models import *
from ICDHierarchyParser import *
from hierarchical_evaluation import *
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

from collections import defaultdict
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

ALL_MODELS = ['Bert', 'XLMRoberta']

def generate_output_dir(args, to_append=None):
    encoder = 'mlbertB' if args.encoder_name_or_path == 'bert-base-multilingual-cased' else None
    encoder = 'mlbertL' if args.encoder_name_or_path == 'bert-large-multilingual-cased' else encoder
    encoder = 'xlmrB' if args.encoder_name_or_path == 'xlm-roberta-base' else encoder
    max_cluster_size = None if 'mc2c' not in args.model else args.max_cluster_size
    min_cluster_size = None if 'mc2c' not in args.model else args.min_cluster_size
    max_cluster_threshold = None if 'mc2c' not in args.model else args.max_cluster_threshold
    max_m = None if 'mc2c' not in args.model else args.max_m
    label_msl = None if args.model != 'la_mc2c' and args.model != 'label_attn' else args.label_max_seq_length
    with_none = 'Without_None_Label' if not args.train_with_none else 'With_None_Label'
    mcc_loss = None if 'mc2c' not in args.model else args.mcc_loss
    model = args.model + '_no_mlcc' if 'mc2c' in args.model and args.no_mlcc else args.model
    lmbda = None if args.lmbda == 1.0 else args.lmbda
    frz_bert = args.n_bert2freeze if args.n_bert2freeze else args.freeze_bert

    output_dir = os.path.join('exps_dir', with_none, model, '_'.join([args.data_dir.split('/')[1],
                                                                str(args.doc_max_seq_length),
                                                                str(label_msl),
                                                                encoder,
                                                                str(args.learning_rate),
                                                                args.loss_fct,
                                                                str(max_cluster_size),
                                                                str(max_cluster_threshold),
                                                                str(min_cluster_size),
                                                                str(max_m),
                                                                str(args.n_gpu),
                                                                str(args.num_train_epochs),
                                                                str(args.per_gpu_train_batch_size),
                                                                str(mcc_loss),
                                                                str(lmbda),
                                                                str(frz_bert)]))
    if to_append:
        output_dir += to_append
    return output_dir


class ICDDataloader(Dataset):
    def __init__(self, data_path):
        self.data = pickle_load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        # return self.data.iloc[idx,]


def plackett_luce(some_list):
    for i in range(1, len(some_list)):
        some_list[i] /= np.sum(some_list[i:])
    return np.sum(np.log(some_list))


def simple_accuracy(preds, labels):

    return (preds == labels).mean()


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def acc_and_f1(preds, labels, metric_avg):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=metric_avg)
    prec = precision_score(y_true=labels, y_pred=preds, average=metric_avg)
    recall = recall_score(y_true=labels, y_pred=preds, average=metric_avg)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "precision": prec,
        "recall": recall,
    }


# MODEL_CLASSES = {
#     "xlmroberta-label_attn": (XLMRobertaConfig, BertForMLSCWithLabelAttention, XLMRobertaTokenizer),
#     "bert-label_attn": (BertConfig, BertForMLSCWithLabelAttention, BertTokenizer),
#     "xlmroberta-stacked": (XLMRobertaConfig, StackedBertForMultiLabelSequenceClassification, XLMRobertaTokenizer),
#     "bert-stacked": (BertConfig, StackedBertForMultiLabelSequenceClassification, BertTokenizer),
#     "xlmroberta-label_attn-stacked": (XLMRobertaConfig, StackedBertForMLSCWithLabelAttention, XLMRobertaTokenizer),
#     "bert-label_attn-stacked": (BertConfig, StackedBertForMLSCWithLabelAttention, BertTokenizer),
#     "xlmroberta-baseline": (XLMRobertaConfig, BertForMultiLabelSequenceClassification, XLMRobertaTokenizer),
#     "bert-baseline": (BertConfig, BertForMultiLabelSequenceClassification, BertTokenizer),
#
# }

MODEL_CLASSES = {
    "bert-label_attn": (BertConfig, BertForMLSCWithLabelAttention, BertTokenizer),
    "bert-baseline": (BertConfig, BertForMultiLabelSequenceClassification, BertTokenizer),
    "bert-mc2c": (BertConfig, MC2C, BertTokenizer),
    "bert-la_mc2c": (BertConfig, LabelAttentionMC2C, BertTokenizer),
    "bert-mc2c-no_mlcc": (BertConfig, MC2C_noMLCC, BertTokenizer),
    # "bert-la_mc2c-no_mlcc": (BertConfig, LabelAttentionMC2C_noMLCC, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    def load_local_labels(self, seed):
        return pickle.load(open(os.path.join(self.args.data_dir,
                              'MCC/{}_{}_{}/{}/train_doc_id2gold.p'.format(self.args.min_cluster_size, self.args.max_cluster_size,
                                                    self.args.max_cluster_threshold, seed)), 'rb'))


def train(args, train_dataset, label_dataset, model, tokenizer, class_weights, idx2id):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    if args.doc_batching:
        train_dataloader = DataLoader(train_dataset, sampler=None, batch_size=args.n_gpu, collate_fn=my_collate)
        train_dataloader = list(train_dataloader)
    else:
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    label_dataloader = DataLoader(label_dataset, sampler=None, batch_size=len(label_dataset))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    p_count = 0
    np_count = 0
    for param in model.parameters():
        if param.requires_grad:
            p_count += 1

    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(name)
            np_count += 1



    # num_warmup_steps = int(len(train_dataloader) * args.warmup_proportion) * args.num_train_epochs
    num_warmup_steps = int(len(train_dataloader) * args.warmup_proportion * args.num_train_epochs)
    # if 'checkpoint-' in args.encoder_name_or_path:
    #     optimizer = torch.load(os.path.join(args.output_dir, 'optimizer.pt'))
    #     scheduler = torch.load(os.path.join(args.output_dir, 'scheduler.pt'))
    # else:
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.encoder_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.encoder_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.encoder_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.encoder_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.model == 'label_attn' or args.model == 'la_mc2c':
        model.initialize_label_data(next(iter(label_dataloader)))

    if 'mc2c' in args.model:
        model.get_idx2id(idx2id)

    # multi-gpu training (should be after apex fp16 initialization)
    n_clusters = model.n_clusters if 'mc2c' in args.model else 0
    if args.n_gpu > 1:
        if args.doc_batching:
            model = MyDataParallel(model)
        else:
            model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num labels = %d", args.num_labels)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    if 'mc2c' in args.model:
        logger.info("  Num Clusters = %d", n_clusters)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.encoder_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.encoder_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    tr_cluster_loss, logging_cluster_loss = 0.0, 0.0
    tr_micro_loss, logging_micro_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility

    for ep, _ in enumerate(train_iterator):
        if args.doc_batching:
            random.shuffle(train_dataloader)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # label_data = next(iter(label_dataloader))
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            if args.doc_batching:
                batch = tuple(tuple(ti.to(args.device) for ti in t) for t in batch)
            else:
                batch = tuple(t.to(args.device) for t in batch)

            inputs = {"doc_input_ids": batch[0], "doc_attention_mask": batch[1], "labels": batch[2], "ranks": batch[4],
                      "epoch": ep, 'doc_ids': batch[3], 'train': True, 't_total':t_total}

            if args.encoder_type == 'bert':
                inputs['token_type_ids'] = batch[-1]
            # outputs = model(**inputs)
            try:
                outputs = model(**inputs)
            except:
                inputs = {"doc_input_ids": batch[0], "doc_attention_mask": batch[1], "labels": batch[2],
                          "ranks": batch[4], "epoch": ep, 'doc_ids': batch[3], 'train': True, 'debug':True}
                outputs = model(**inputs)
#
            if 'mc2c' in args.model and not args.no_mlcc:
                cluster_loss, micro_loss = outputs[0], outputs[1]
                micro_loss = args.lmbda * micro_loss
                loss = cluster_loss + micro_loss

                # cluster_loss, micro_loss, loss  = outputs[0], outputs[1], outputs[2]


            elif 'mc2c' in args.model and args.no_mlcc:
                cluster_loss = torch.Tensor([0])
                micro_loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                loss = micro_loss
            else:
                cluster_loss, micro_loss = torch.Tensor([0]), torch.Tensor([0])
                loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                cluster_loss = cluster_loss.mean()
                micro_loss = micro_loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                cluster_loss = cluster_loss / args.gradient_accumulation_steps
                micro_loss = micro_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            tr_cluster_loss += cluster_loss.item()
            tr_micro_loss += micro_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        if 'mc2c' in args.model:
                            results = evaluate_mc2c(args, model, tokenizer)
                        else:
                            results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    cluster_loss_scalar = (tr_cluster_loss - logging_cluster_loss) / args.logging_steps
                    micro_loss_scalar = (tr_micro_loss - logging_micro_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logs['cluster loss'] = cluster_loss_scalar
                    logs['micro loss'] = micro_loss_scalar

                    logging_loss = tr_loss
                    logging_cluster_loss = tr_cluster_loss
                    logging_micro_loss = tr_micro_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    # print(torch.sum(model.cluster_classifier.weight))

    return global_step, tr_loss / global_step


def evaluate_mc2c(args, model, tokenizer, prefix="", test=False):
    eval_output_dir = args.output_dir
    # print(torch.sum(model.cluster_classifier.weight))

    results = {}

    eval_dataset, label_dataset, idx2id = load_and_cache_examples(args, tokenizer, evaluate=True, label_data=True) if not test else load_and_cache_examples(args, tokenizer, test=True, label_data=True)
    if 'mc2c' in args.model:
        model.get_idx2id(idx2id)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)

    if args.doc_batching:
        eval_dataloader = DataLoader(eval_dataset, sampler=None, batch_size=1, collate_fn=my_collate)
    else:
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    label_dataloader = DataLoader(label_dataset, sampler=None, batch_size=len(label_dataset))
    if args.model == 'label_attn' or args.model == 'la_mc2c':
        model.initialize_label_data(next(iter(label_dataloader)))

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    ids = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        if args.doc_batching:
            batch = tuple(tuple(ti.to(args.device) for ti in t) for t in batch)
        else:
            batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():

            ##############################
            if args.doc_batching:
                input_ids = batch[0][0]
                attn_mask = batch[1][0]
                labels = batch[2][0]
                # ranks = batch[4][0]
            else:
                input_ids = batch[0]  # may need to fix this!
                attn_mask = batch[1]  # may need to fix this!
                labels = batch[2]
                # ranks = batch[4]

            inputs = {"doc_input_ids": input_ids, "doc_attention_mask": attn_mask, "labels": labels, "ranks": None, 'doc_ids': batch[3]}
            if args.encoder_type == 'bert':
                inputs['token_type_ids'] = batch[-1][0] # prolly gonna need to fix this

            #############################
            logits = model(**inputs)[0]
            tmp_ids = []
            for doc_id, logit in logits.items():
                n_labels = logit.shape[0]
                ids.append(doc_id)
                tmp_ids.append(doc_id)

            logits = torch.cat([logits[d] for d in tmp_ids])
            logits.reshape((-1, n_labels))

            eval_loss = 0
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            if args.doc_batching:
                out_label_ids = batch[2][0].detach().cpu().numpy()
            else:
                out_label_ids = batch[2].detach().cpu().numpy()

        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            if args.doc_batching:
                out_label_ids = np.append(out_label_ids, batch[2][0].detach().cpu().numpy(), axis=0)
            else:
                out_label_ids = np.append(out_label_ids, batch[2].detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps
    preds = preds.reshape((len(eval_dataset), args.num_labels))


    if args.train_with_none:
        out_label_ids = out_label_ids.reshape((len(eval_dataset), args.num_labels-1))
    else:
        out_label_ids = out_label_ids.reshape((len(eval_dataset), args.num_labels))
    if args.train_with_none:
        preds =  preds[:,:-1]

    total_uniq = len(np.nonzero(np.sum(preds, axis=0))[0])
    total_uniq_true = len(np.nonzero(np.sum(out_label_ids, axis=0))[0])

    sorted_preds_idx = np.flip(np.argsort(preds), axis=1)


    preds = (preds > args.prediction_threshold)

    if not args.train_with_none:
        assert preds.shape == out_label_ids.shape


    result = acc_and_f1(preds, out_label_ids, args.metric_avg)
    results.update(result)

    n_labels = np.sum(preds, axis=1)
    avg_pred_n_labels = np.mean(n_labels)
    avg_true_n_labels = np.mean(np.sum(out_label_ids, axis=1))
    labels_in_preds_not_in_gold = set(np.nonzero(preds)[1]) - set(np.nonzero(out_label_ids)[1])
    labels_in_gold_not_in_preds = set(np.nonzero(out_label_ids)[1]) - set(np.nonzero(preds)[1])
    preds = np.array([sorted_preds_idx[i, :n] for i, n in enumerate(n_labels)])


    with open(os.path.join(args.data_dir, "mlb_{}_{}.p".format(args.label_threshold, args.train_on_all)),
              "rb") as rf:
        mlb = pickle.load(rf)

    preds = [mlb.classes_[preds[i][:]].tolist() for i in range(preds.shape[0])]
    id2preds = {val: preds[i] for i, val in enumerate(ids)}
    preds = [id2preds[val] if val in id2preds else [] for i, val in enumerate(ids)]
    out_label_ids = [mlb.classes_[out_label_ids.astype(int)[i, :].astype(bool)].tolist() for i in
             range(out_label_ids.shape[0])]
    id2gold = {val: out_label_ids[i] for i, val in enumerate(ids)}
    out_label_ids = [id2gold[val] if val in id2gold else [] for i, val in enumerate(ids)]

    with open(os.path.join(args.output_dir, f"preds_{'dev' if not test else 'test'}.tsv"), "w") as wf, \
            open(os.path.join(args.output_dir, f"gold_{'test' if test else 'dev'}.tsv"), "w") as gf, \
            open(os.path.join(args.output_dir, f"preds_{'test' if test else 'dev'}2.tsv"), "w") as pf:
        wf.write("file\tcode\n")
        gf.write("file\tcode\n")
        for idx, doc_id in enumerate(ids):
            pf.write(str(doc_id) + "\t" + '|'.join(preds[idx]) + "\n")
            for p in preds[idx]:
                if p != 'None':
                    line = str(doc_id) + "\t" + p + "\n"
                    wf.write(line)
            for g in out_label_ids[idx]:
                if g != 'None':
                    line = str(doc_id) + "\t" + g + "\n"
                    gf.write(line)

    if 'cantemist' in args.data_dir:
        eval_cmd = [f'python cantemist-evaluation-library/src/main.py -g ' \
                   f'data/cantemist/{"test-set/cantemist-coding/test-coding.tsv" if test else "dev-set1/cantemist-coding/dev1-coding.tsv"} -p ' \
                   f'{args.output_dir}/preds_{"dev" if not test else "test"}.tsv ' \
                   f'-c cantemist-evaluation-library/valid-codes.tsv -s coding',
                   # f'python cantemist-evaluation-library/src/comp_f1_diag_proc.py '
                   # f'-g 'f'data/cantemist/{"test-set/cantemist-coding/test-coding.tsv" if test else "dev-set1/cantemist-coding/dev1-coding.tsv"} '
                   # f'-p {args.output_dir}/preds_{"dev" if not test else "test"}.tsv '
                   # f'-c cantemist-evaluation-library/valid-codes.tsv '
                   # f'-f  data/cantemist/{"test-set/valid_files.txt" if test else "dev-set1/valid_files.txt"} '
                    ]
    elif 'spanish' in args.data_dir:
        eval_cmd = [f'python codiesp-evaluation-script/comp_f1_diag_proc.py '
                    f'-g data/Spanish/final_dataset_v4_to_publish/{"test/testD.tsv" if test else "dev/devD.tsv"} '
                    f'-p {args.output_dir}/preds_{"dev" if not test else "test"}.tsv  '
                    f'-c codiesp-evaluation-script/codiesp_codes/codiesp-D_codes.tsv']
    else:
        eval_cmd = [f"python evaluation.py --ids_file=data/German/"
                    f"{'nts_icd_test/ids_test.txt' if test else 'nts-icd_train/ids_development.txt'} \
                     --anns_file=data/German/"
                    f"{'nts_icd_test/anns_test.txt' if test else 'nts-icd_train/anns_train_dev.txt'} \
                     --dev_file={args.output_dir}/preds_{'test' if test else 'dev'}2.tsv \
                     --out_file={args.output_dir}/official_eval_results.txt"]

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    if args.eval_cluster_activator:
        cluster_activator_f1 = np.mean(model.cluster_activator_f1)
        cluster_activator_p = np.mean(model.cluster_activator_p)
        cluster_activator_r = np.mean(model.cluster_activator_r)
    else:
        cluster_activator_f1 = None

    ##########################################################################################################
    ##                                          Hierarchical Eval                                          ##
    ##########################################################################################################
    hierarchical_evaluator = HierarchicalEvaluator(args, test=test, reduced=True)
    hier_eval_results = hierarchical_evaluator.do_hierarchical_eval()
    ##########################################################################################################
    ##########################################################################################################

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} - Zero-Shot Labels Removed *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        for e in eval_cmd:
            eval_results = os.popen(e).read()
            print("*** Eval results with challenge script: *** ")
            print(eval_results)
            writer.write(eval_results)
        temp = "Average #labels/doc preds: " + str(avg_pred_n_labels) + \
               "\nAverage #labels/doc true: " + str(avg_true_n_labels) +  \
               "\nTotal unique labels predicted: " + str(total_uniq) + \
               "\nTotal unique labels true: " + str(total_uniq_true) + \
               "\nNumber of unique labels in preds which are not in gold: " + str(len(labels_in_preds_not_in_gold)) + \
               "\nNumber of unique labels in gold which are not in preds: " + str(len(labels_in_gold_not_in_preds))

        if cluster_activator_f1:
            temp += '\nCluster activator F1:' + str(cluster_activator_f1) + \
                    '\nCluster activator P:' + str(cluster_activator_p) + \
                    '\nCluster activator R:' + str(cluster_activator_r)

        writer.write(temp)
        print(temp)

        print("***** Hierarchical eval results - Zero-Shot Labels Removed: ***** ")
        writer.write(hier_eval_results)
        print(hier_eval_results)

    print("\n\nOutput dir: ", args.output_dir)
    print("Number clusters: ", args.n_clusters)
    if args.eval_full:
        eval_on_all(args, idx2id, mlb, test)

    return results


def evaluate(args, model, tokenizer, prefix="", test=False):
    eval_output_dir = args.output_dir

    results = {}

    eval_dataset, label_dataset, idx2id = load_and_cache_examples(args, tokenizer, evaluate=True,
                                                                  label_data=True) if not test else load_and_cache_examples(
        args, tokenizer, test=True, label_data=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)

    if args.doc_batching:
        eval_dataloader = DataLoader(eval_dataset, sampler=None, batch_size=1, collate_fn=my_collate)
    else:
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    label_dataloader = DataLoader(label_dataset, sampler=None, batch_size=len(label_dataset))
    if args.model == 'label_attn' or args.model == 'la_mc2c':
        model.initialize_label_data(next(iter(label_dataloader)))

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    ids = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # label_data = next(iter(label_dataloader))
        model.eval()
        # batch = tuple(t.to(args.device) for t in batch)

        if args.doc_batching:
            batch = tuple(tuple(ti.to(args.device) for ti in t) for t in batch)
        else:
            batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():

            ##############################
            if args.doc_batching:
                input_ids = batch[0][0]
                attn_mask = batch[1][0]
                labels = batch[2][0]
                ranks = batch[4][0]
            else:
                input_ids = batch[0]  # may need to fix this!
                attn_mask = batch[1]  # may need to fix this!
                labels = batch[2]
                ranks = batch[4]
            inputs = {"doc_input_ids": input_ids, "doc_attention_mask": attn_mask, "labels": labels, "ranks": ranks}
            if args.encoder_type == 'bert':
                inputs['token_type_ids'] = batch[-1][0] # prolly gonna need to fix this

            #############################
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        # doc_dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels,
        #                                         all_doc_ids, all_label_ranks, all_token_type_ids)
        if preds is None:
            # preds = logits.detach().cpu().numpy()
            preds = logits.detach().cpu().numpy()
            if args.doc_batching:
                out_label_ids = batch[2][0].detach().cpu().numpy()
            else:
                out_label_ids = batch[2].detach().cpu().numpy()

        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            # preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            # print(len(preds))
            if args.doc_batching:
                out_label_ids = np.append(out_label_ids, batch[2][0].detach().cpu().numpy(), axis=0)
            else:
                out_label_ids = np.append(out_label_ids, batch[2].detach().cpu().numpy(), axis=0)

        if len(ids) == 0:
            if args.doc_batching:
                ids.append(batch[3][0].detach().cpu().numpy().item())
            else:
                ids.append(batch[3].detach().cpu().numpy())

        else:
            if args.doc_batching:
                ids.append(batch[3][0].detach().cpu().numpy().item())
            else:
                ids[0] = np.append(
                    ids[0], batch[3].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds.reshape((len(eval_dataset), args.num_labels))

    out_label_ids = out_label_ids.reshape((len(eval_dataset), args.num_labels))

    preds = sigmoid(preds)

    preds[preds < args.prediction_threshold] = 0

    total_uniq = len(np.nonzero(np.sum(preds, axis=0))[0])
    total_uniq_true = len(np.nonzero(np.sum(out_label_ids, axis=0))[0])

    sorted_preds_idx = np.flip(np.argsort(preds), axis=1)
    preds = (preds > args.prediction_threshold)

    assert preds.shape == out_label_ids.shape

    result = acc_and_f1(preds, out_label_ids, args.metric_avg)
    results.update(result)

    n_labels = np.sum(preds, axis=1)
    avg_pred_n_labels = np.mean(n_labels)
    avg_true_n_labels = np.mean(np.sum(out_label_ids, axis=1))
    labels_in_preds_not_in_gold = set(np.nonzero(preds)[1]) - set(np.nonzero(out_label_ids)[1])
    labels_in_gold_not_in_preds = set(np.nonzero(out_label_ids)[1]) - set(np.nonzero(preds)[1])
    preds = np.array([sorted_preds_idx[i, :n] for i, n in enumerate(n_labels)])

    # preds = np.array(sorted_preds_idx[:n_labels])

    if not args.doc_batching:
        ids = ids[0]
    # ids = np.array([i for i in range(ids[-1]+1)])

    with open(os.path.join(args.data_dir, "mlb_{}_{}.p".format(args.label_threshold, args.train_on_all)),
              "rb") as rf:
        mlb = pickle.load(rf)
    # preds = [mlb.classes_[preds[i, :].astype(bool)].tolist() for i in range(preds.shape[0])]

    preds = [mlb.classes_[preds[i][:]].tolist() for i in range(preds.shape[0])]
    # preds = mlb.classes_[preds[:]].tolist()

    id2preds = {val: preds[i] for i, val in enumerate(ids)}
    preds = [id2preds[val] if val in id2preds else [] for i, val in enumerate(ids)]

    out_label_ids = [mlb.classes_[out_label_ids.astype(int)[i, :].astype(bool)].tolist() for i in
             range(out_label_ids.shape[0])]
    id2gold = {val: out_label_ids[i] for i, val in enumerate(ids)}
    out_label_ids = [id2gold[val] if val in id2gold else [] for i, val in enumerate(ids)]

    with open(os.path.join(args.output_dir, f"preds_{'dev' if not test else 'test'}.tsv"), "w") as wf, \
            open(os.path.join(args.output_dir, f"gold_{'test' if test else 'dev'}.tsv"), "w") as gf, \
            open(os.path.join(args.output_dir, f"preds_{'test' if test else 'dev'}2.tsv"), "w") as pf:
        wf.write("file\tcode\n")
        gf.write("file\tcode\n")
        for idx, doc_id in enumerate(ids):
            pf.write(str(idx2id[doc_id]) + "\t" + '|'.join(preds[idx]) + "\n")
            for p in preds[idx]:
                if p != 'None':
                    line = str(idx2id[doc_id]) + "\t" + p + "\n"
                    wf.write(line)
            for g in out_label_ids[idx]:
                if g != 'None':
                    line = str(idx2id[doc_id]) + "\t" + g + "\n"
                    gf.write(line)


    if 'cantemist' in args.data_dir:
        eval_cmd = f'python cantemist-evaluation-library/src/main.py -g ' \
                   f'data/cantemist/{"test-set/cantemist-coding/test-coding.tsv" if test else "dev-set1/cantemist-coding/dev1-coding.tsv"} -p ' \
                   f'{args.output_dir}/preds_{"dev" if not test else "test"}.tsv ' \
                   f'-c cantemist-evaluation-library/valid-codes.tsv -s coding'
    elif 'spanish' in args.data_dir:
    	eval_cmd = f'python codiesp-evaluation-script/comp_f1_diag_proc.py ' \
                   f'-g data/Spanish/final_dataset_v4_to_publish/{"test/testD.tsv" if test else "dev/devD.tsv"} ' \
                   f'-p {args.output_dir}/preds_{"dev" if not test else "test"}.tsv  ' \
                   f'-c codiesp-evaluation-script/codiesp_codes/codiesp-D_codes.tsv'
    else:
        eval_cmd = f"python evaluation.py --ids_file=data/German/{'nts_icd_test/ids_test.txt' if test else 'nts-icd_train/ids_development.txt'} \
                     --anns_file=data/German/{'nts_icd_test/anns_test.txt' if test else 'nts-icd_train/anns_train_dev.txt'} \
                     --dev_file={args.output_dir}/preds_{'test' if test else 'dev'}2.tsv \
                     --out_file={args.output_dir}/official_eval_results.txt"

    ##########################################################################################################
    ##                                          Hierarchical Eval                                          ##
    ##########################################################################################################
    hierarchical_evaluator = HierarchicalEvaluator(args, test=test, reduced=True)
    hier_eval_results = hierarchical_evaluator.do_hierarchical_eval()
    ##########################################################################################################
    ##########################################################################################################

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} - Zero-Shot Labels Removed *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        eval_results = os.popen(eval_cmd).read()
        print("*** Eval results with challenge script: *** ")
        print(eval_results)
        writer.write(eval_results)
        temp = "Average #labels/doc preds: " + str(avg_pred_n_labels) + \
               "\nAverage #labels/doc true: " + str(avg_true_n_labels) +  \
               "\nTotal unique labels predicted: " + str(total_uniq) + \
               "\nTotal unique labels true: " + str(total_uniq_true) + \
               "\nNumber of unique labels in preds which are not in gold: " + str(len(labels_in_preds_not_in_gold)) + \
               "\nNumber of unique labels in gold which are not in preds: " + str(len(labels_in_gold_not_in_preds))
        writer.write(temp)
        print(temp)

        print("\n\n***** Hierarchical eval results - Zero-Shot Labels Removed: ***** ")
        writer.write(hier_eval_results)
        print(hier_eval_results)

    print("Output dir: ", args.output_dir)

    eval_on_all(args, idx2id, mlb, test)

    return results


def eval_on_all(args, idx2id, mlb, testing=False):
    hierarchical_evaluator = HierarchicalEvaluator(args, test=testing)
    hier_eval_results = hierarchical_evaluator.do_hierarchical_eval()
    print("\n\n***** Hierarchical eval results on ALL labels : ***** ")
    print(hier_eval_results)
    def load_gold_data():
        path2gold = os.path.join(args.data_dir, f"{'test' if testing else 'dev'}_{args.label_threshold}_{args.ignore_labelless_docs}.tsv")
        gold = [d.split('\t') for d in open(path2gold, 'r').read().splitlines()[1:]]
        gold = [[d[0], d[2]] for d in gold]
        return gold
    with open(os.path.join(args.output_dir, f"preds_{'test' if testing else 'dev'}.tsv"), 'r') as tf:
        test_preds = tf.read().splitlines()
    test, gold = defaultdict(list), defaultdict(list)
    all_labels = set(mlb.classes_)
    for line in test_preds[1:]:
        doc_id, label = line.split('\t')
        test[doc_id].append(label)
        all_labels.add(label)
    for doc_id, labels in load_gold_data():
        labels = labels.split('|')
        gold[doc_id] = labels
        all_labels = all_labels.union(set(labels))
    mlb = MultiLabelBinarizer()
    mlb.fit([all_labels])
    test_preds, gold_labels = [], []
    for doc_id in set(test.keys()).union(set(gold.keys())):
        test_preds.append(mlb.transform([test[doc_id]])[0] if test[doc_id] else np.zeros(len(mlb.classes_)))
        gold_labels.append(mlb.transform([gold[doc_id]])[0] if gold[doc_id] else np.zeros(len(mlb.classes_)))
    test_preds, gold_labels = np.array(test_preds), np.array(gold_labels)
    result = acc_and_f1(test_preds, gold_labels, 'micro')
    logger.info("***** Eval results on All Labels *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    print('gold_labels.shape'.upper(), gold_labels.shape)


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        # required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--encoder_type",
        default=None,
        type=str,
        # required=True,
        help="Encoder type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--encoder_name_or_path",
        default=None,
        type=str,
        # required=True,
        help="Path to pre-trained encoder or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )

    parser.add_argument(
        "--model",
        default=None,
        type=str,
        # required=True,
        help="Which model to use for experiments: baseline, label_attention, mc2c, la_mc2c"
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--prediction_threshold",
        default=0.5,
        type=float,
        help="Threshold at which to decide between 0 and 1 for labels.",
    )
    parser.add_argument("--loss_fct", default="none", type=str, help="The function to use.")
    parser.add_argument("--mcc_loss", default="ldam", type=str, help="The multi class loss function to use.")
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as encoder_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as encoder_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--doc_max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--label_max_seq_length",
        default=15,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--lmbda", type=float, default=1.0, help="How much to scale down MCC losses. ")
    parser.add_argument("--max_hierarchical_distance", type=int, default=100000,
                        help="specifies the maximum distance that the measures will search in order "
                             "to link nodes. Above that threshold all nodes will be considered to have a "
                             "common ancestor. For example if a value of 1 is used then all nodes are considered "
                             "to have a dummy common ancestor as direct parent of them. This option should "
                             "usually be set to a very large number (for example 100000). But in very large "
                             "datasets it should be set to values like 2 or 3 for computational reasons (see "
                             "paper for further details).")
    parser.add_argument("--max_hierarchical_error", type=int, default=5,
                        help="specifies the maximum error with which pair-based measures penalize"
                             "nodes that were matched with a default one (see paper for further details).")
    parser.add_argument("--label_threshold", type=int, default=0, help="Exclude labels which occur <= threshold")
    parser.add_argument("--min_cluster_size", type=int, default=5, help="Smallest allowed cluster size.")
    parser.add_argument("--max_cluster_size", type=int, default=10, help="Largest allowed cluster size.")
    parser.add_argument("--max_cluster_threshold", type=float, default=.25, help="Largest relative label frequency allowed"
                                                                               "into cluster.")
    parser.add_argument("--rakel", type=str, default=None, help="O for 'overlapping', 'D' for distinct")
    parser.add_argument("--logit_aggregation", type=str, default='max', help="Whether to aggregate logits by max value "
                                                                             "or average value. Options:"
                                                                             "'--max', '--avg'")
    parser.add_argument("--hierarchical_clustering", action="store_true", help="Whether to perform clustering based on "
                                                                               "hierarchical distance.")
    parser.add_argument("--preprocess", action="store_true", help="Whether to do the initial processing of the data.")
    parser.add_argument("--train_on_all", action="store_true", help="Whether to train on train + dev for final testing.")
    parser.add_argument("--ignore_labelless_docs", action="store_true",
                        help="Whether to ignore the documents which have no labels.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--train_with_none", action="store_true", help="Whether to add 'None' label to clusters")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run testing.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument("--n_clusters", default=None, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--make_plots', action='store_true', help="Whether to make plots on data.")
    parser.add_argument('--eval_full', action='store_true', help="Whether to evaluate the model on the full labelset.")
    parser.add_argument('--freeze_bert', action='store_true', help="Whether to freeze the BERT encoder.")
    parser.add_argument('--n_bert2freeze', default=None, type=int, help="How many bert layers to freeze.")
    parser.add_argument('--no_mlcc', action='store_true', help="Whether to train MLCC in MC2C.")
    parser.add_argument('--eval_all_labels', action='store_true', help="Whether to evaluate on all labels"
                                                                       "or only those which occur in training data.")
    parser.add_argument('--eval_cluster_activator', action='store_true', help="Evaluate performance of cluster activation classifier")
    parser.add_argument('--do_iterative_class_weights', action='store_true', help="Whether to use iteratively "
                                                                                  "calculated class weights")
    parser.add_argument('--use_bce_class_weights', action='store_true', help='If using BCE for MLCC/main loss, whether to'
                                                                              'use class weights for it.')
    parser.add_argument('--use_mcc_class_weights', action='store_true', help='If using BCE for MCC loss, whether to'
                                                                              'use class weights for it.')
    parser.add_argument('--pass_mlcc_preds_to_mccs', action='store_true', help="Whether to pass the other predictions "
                                                                               "on the activated clusters to the MCC classifiers.")
    parser.add_argument('--mlcc_as_gates', action='store_true', help='Whether to use MLCC preds as '
                                                                     'weights for MCC losses.')
    parser.add_argument('--DRW', action='store_true', help="Whether to do deferred reweighting.")
    parser.add_argument('--doc_batching', action='store_true', help="Whether to fit one document into a batch during")
    parser.add_argument("--metric_avg", default='micro', type=str, help="Micro vs macro for F1/P/R")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_m", default=0.5, type=float, help="Max margin for LDAMLoss.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Linear warmup over warmup proportion.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as encoder_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=21, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    args = parser.parse_args()

    try:
        out_dir = args.output_dir
        old_args = torch.load(os.path.join(args.output_dir, 'training_args.bin'))
        old_args_dict = {}
        for arg in vars(old_args):
            old_args_dict['args.'+str(arg)] = getattr(old_args, arg)
        for k, v in old_args_dict.items():
            try:
                exec("%s=%s" % (k, v))
            except:
                exec("%s='%s'" % (k, v))
        args.output_dir = out_dir
        args.do_train = False
        args.do_eval = False
        args.do_test = True
    except:
        pass

    if args.no_mlcc:
        args.train_with_none = True
    if args.n_bert2freeze:
        args.freeze_bert = True
    # if args.doc_batching:
    #     args.per_gpu_train_batch_size = 10
    #     args.per_gpu_eval_batch_size = 10

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    if not args.output_dir:
        args.output_dir = generate_output_dir(args)
    elif args.output_dir[0] == '^':
        args.output_dir = generate_output_dir(args, to_append=args.output_dir)

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    if args.preprocess:
        if 'cantemist' in args.data_dir:
            reader = CantemistReader(args)
        elif 'german' in args.data_dir:
            reader = GermanReader(args)
        elif 'spanish' in args.data_dir:
            reader = SpanishReader(args)
        else:
            print("Problem with data directory.")
        args.overwrite_cache = True

    # Prepare task
    try:
        processor = MyProcessor(args)
    except:
        if 'cantemist' in args.data_dir:
            reader = CantemistReader(args)
        elif 'german' in args.data_dir:
            reader = GermanReader(args)
        elif 'spanish' in args.data_dir:
            reader = SpanishReader(args)
        reader.process_data()
        processor = MyProcessor(args)
    if 'spanish' in args.data_dir:
        gen = SpanishICD10Hierarchy(args)
    elif 'german' in args.data_dir:
        gen = GermanICD10Hierarchy(args)
    elif 'cantemist' in args.data_dir:
        gen = CantemistICD10Hierarchy(args)
    class_weights = processor.get_class_counts()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.num_labels = num_labels + 1 if args.train_with_none else num_labels
    if 'mc2c' in args.model:
        cluster_out_dir = os.path.join(args.data_dir,
                                       'MCC/{}_{}_{}_{}'.format(args.min_cluster_size, args.max_cluster_size,
                                                                args.max_cluster_threshold, args.train_with_none))
        if args.hierarchical_clustering and not args.train_with_none:
            clusterer = MC2CHierarchicalLabelClusterer(args.data_dir, max_cluster_size=args.max_cluster_size,
                                        min_cluster_size=args.min_cluster_size,
                                        max_freq_threshold=args.max_cluster_threshold, add_none=False)
            cluster_out_dir = os.path.join(args.data_dir,
                         'MCC/{}_{}_{}_{}_Hierarchical_Clustering'.format(args.min_cluster_size,
                                                                          args.max_cluster_size,
                                                                          args.max_cluster_threshold,
                                                                          args.train_with_none))
        elif args.train_with_none and not args.hierarchical_clustering:
            clusterer = MC2CLabelClusterer_None(args.data_dir, max_cluster_size=args.max_cluster_size,
                                    min_cluster_size=args.min_cluster_size,
                                    max_freq_threshold=args.max_cluster_threshold, add_none=True)
        elif args.train_with_none and args.hierarchical_clustering:
            clusterer = MC2CHierarchicalLabelClusterer_None(args.data_dir, max_cluster_size=args.max_cluster_size,
                                    min_cluster_size=args.min_cluster_size,
                                    max_freq_threshold=args.max_cluster_threshold, add_none=True)
            cluster_out_dir = os.path.join(args.data_dir,
                         'MCC/{}_{}_{}_{}_Hierarchical_Clustering'.format(args.min_cluster_size,
                                                                          args.max_cluster_size,
                                                                          args.max_cluster_threshold,
                                                                          args.train_with_none))
        else:
            clusterer = MC2CLabelClusterer(args.data_dir, max_cluster_size=args.max_cluster_size,
                                        min_cluster_size=args.min_cluster_size,
                                        max_freq_threshold=args.max_cluster_threshold, add_none=False)


        clusterer.main()

        # vv for looking up the labels in the "predict activated clusters" phase
        doc_ids2_clusters = pickle.load(open(os.path.join(cluster_out_dir, 'doc_ids2clusters.p'), 'rb'))
        cluster_idx2seed = pickle.load(open(os.path.join(cluster_out_dir, 'cluster_idx2seed.p'), 'rb'))
        args.n_clusters = len(clusterer.clusters)


    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.encoder_type = args.encoder_type.lower()

    model_name = args.encoder_type + '-' + args.model
    model_name = model_name + '-no_mlcc' if 'mc2c' in model_name and args.no_mlcc else model_name


    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_name]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.encoder_name_or_path,
        num_labels=num_labels,
        finetuning_task="thesis",
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.encoder_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # Training
    if args.do_train:
        try:
            model = model_class.from_pretrained(
                args.encoder_name_or_path,
                from_tf=bool(".ckpt" in args.encoder_name_or_path),
                config=config,
                loss_fct=args.loss_fct,
                args=args,
                class_weights=class_weights,
                doc_ids2_clusters=doc_ids2_clusters,
                clusters=clusterer.clusters,
                cluster_idx2seed=cluster_idx2seed,
                cluster_output_dir=cluster_out_dir,
            )
        except:
            model = model_class.from_pretrained(
                args.encoder_name_or_path,
                from_tf=bool(".ckpt" in args.encoder_name_or_path),
                config=config,
                loss_fct=args.loss_fct,
                args=args,
                class_weights=class_weights,
            )
        if args.freeze_bert:
            model.freeze_bert_encoder()


        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)

        logger.info("Training/evaluation parameters %s", args)


        train_dataset, label_dataset, idx2id = load_and_cache_examples(args, tokenizer, evaluate=False, label_data=True)
        args.n_examples = len(train_dataset)

        global_step, tr_loss = train(args, train_dataset, label_dataset, model, tokenizer, class_weights, idx2id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval or args.do_test and args.local_rank in [-1, 0]:
        args.eval_full = True if args.do_test else args.eval_full
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            try:
                model = model_class.from_pretrained(
                    checkpoint,
                    loss_fct=args.loss_fct,
                    args=args,
                    doc_ids2_clusters=doc_ids2_clusters,
                    clusters=clusterer.clusters,
                    cluster_idx2seed=cluster_idx2seed,
                    cluster_output_dir=cluster_out_dir,
                )
            except:
                model = model_class.from_pretrained(
                    checkpoint,
                    loss_fct=args.loss_fct,
                    args=args,
                )
            model.to(args.device)
            if 'mc2c' in args.model:
                result = evaluate_mc2c(args, model, tokenizer, prefix=prefix, test=args.do_test)
            else:
                result = evaluate(args, model, tokenizer, prefix=prefix, test=args.do_test)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    # # Make predictions on test set
    # results = {}
    # if args.do_test and args.local_rank in [-1, 0]:
    #     if not os.path.exists(os.path.join(args.output_dir, "preds_test.txt")):
    #         tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    #         checkpoints = [args.output_dir]
    #         if args.eval_all_checkpoints:
    #             checkpoints = list(
    #                 os.path.dirname(c) for c in
    #                 sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
    #             )
    #             logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #         logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #         for checkpoint in checkpoints:
    #             global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
    #             prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
    #             model = model_class.from_pretrained(checkpoint, args=args, loss_fct=args.loss_fct)
    #             model.to(args.device)
    #             predictions = generate_test_preds(args, model, tokenizer, prefix=global_step)
    #     # evaluate_test_preds(args)

    return results


if __name__ == '__main__':
    main()

    """
    Next steps: rewrite predict() function so I can evaluate on the test set, against:
        1) FULL labels
        2) ignoring labelless docs (set a flag for this)
    """

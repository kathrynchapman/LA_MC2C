import pickle
import os
import numpy as np
import logging
import torch
from torch.utils.data import TensorDataset, Dataset
from itertools import chain
import sys

import string

logger = logging.getLogger(__name__)

class BatchedDataset(Dataset):
    def __init__(self, tensors):
        self.tensor0 = tensors[0]
        self.tensor1 = tensors[1]
        self.tensor2 = tensors[2]
        self.tensor3 = tensors[3]
        self.tensor4 = tensors[4]
        try:
            self.tensor5 = tensors[5]
        except:
            self.tensor5 = None

    def __getitem__(self, index):
        if self.tensor5 is not None:
            t = (self.tensor0[index],
                 self.tensor1[index],
                 self.tensor2[index],
                 self.tensor3[index],
                 self.tensor4[index],
                 self.tensor5[index],)
        else:
            t = (self.tensor0[index],
                 self.tensor1[index],
                 self.tensor2[index],
                 self.tensor3[index],
                 self.tensor4[index],)

        return t

    def __len__(self):
        return len(self.tensor0)

def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


class InputExample():
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, labels=None, ranks=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.labels_binary = labels
        # self.label_ranks = ranks


class InputFeatures():
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_ranks = label_ranks
        self.guid = guid


class DataProcessor():
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return pickle_load(input_file)


class MyProcessor(DataProcessor):
    """Processor for German CLEF eHealth 2019 data set."""

    def __init__(self, args, test_file=None):
        self.train_file = os.path.join(args.data_dir,
                                       'train_{}_{}.p'.format(args.label_threshold, args.train_on_all))
        self.dev_file = os.path.join(args.data_dir,
                                     'dev_{}_{}.p'.format(args.label_threshold, args.train_on_all))
        self.test_file = os.path.join(args.data_dir,
                                      'test_{}_{}.p'.format(args.label_threshold, args.train_on_all))
        self.label_desc_file = os.path.join(args.data_dir, "label_desc_{}.p".format(str(args.label_max_seq_length)))
        # label_desc = self.get_label_desc()
        self.mlb = pickle_load(
            os.path.join(args.data_dir, "mlb_{}_{}.p".format(str(args.label_threshold), args.train_on_all)))
        self.class_counts = pickle_load(
            os.path.join(args.data_dir, "class_counts_{}_{}.p".format(str(len(self.mlb.classes_)), args.train_on_all)))

        data = self._read_tsv(self.train_file)
        y = np.array([i[-1] for i in data])
        self.pos_weight = ((1 - y).sum(0) / y.sum(0)).astype(int)

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv(self.train_file))

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv(self.dev_file))

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv(self.test_file))

    def get_label_desc(self):
        return self._create_examples(self._read_tsv(self.label_desc_file), data_type='label_desc')

    def get_labels(self):
        return self.mlb.classes_.tolist()

    def get_class_counts(self):
        return self.class_counts

    def _create_examples(self, data, data_type=''):
        examples = []
        idx2id = {}
        # each d is tuple of ('ID', 'text', binary labels)
        for i, d in enumerate(data):
            guid = d[0]

            if guid == 'id':
                continue
            idx2id[i], guid = guid, i
            text = d[1]
            if not data_type == 'label_desc':
                labels = d[2].tolist()
                examples.append(InputExample(guid=guid, text=text, labels=labels))
            else:
                examples.append(InputExample(guid=guid, text=text))
        return examples, idx2id


def convert_examples_to_features(
        args,
        examples,
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        label_examples=False,
        doc_batching=False,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """

    processor = MyProcessor(args)
    if label_list is None:
        label_list = processor.get_labels()
        logger.info("Using label list %s for task %s" % (label_list, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        try:
            if label_examples or not doc_batching:
                batch = [tokenizer.encode_plus(example.text, add_special_tokens=True, max_length=max_length,)]
            else:
                max_length = max_length - 2
                tokens = tokenizer.tokenize(example.text)
                total_tokens = len(tokens)



                stride = 50
                step = max_length - stride

                n = max_length + (step*(args.per_gpu_train_batch_size-1)) - 1

                tokens = tokens[:n]
                tokens = [tokens[i:i + max_length] for i in range(0, len(tokens), step)]
                tokens = [tokens[0]] + [t for i, t in enumerate(tokens[1:]) if len(tokens[i]) == max_length]

                ################
                # text : 'Today is a lovely day for some NLP!'
                # tokens: ['Today', 'is', 'a', 'love', '##ly', 'day', 'for', 'some', 'NL', '##P', '!']
                # max_length = 5
                # stride = 2
                # step = 3
                # tokens = [['Today', 'is', 'a'], ['is', 'a', 'love'], ['a', 'love', '##ly'], ['love', '##ly', 'day'],
                #               ['##ly', 'day', 'for'], ['day', 'for', 'some'], ['for', 'some', 'NL'],
                #               ['some', 'NL', '##P'], ['NL', '##P', '!'], ['##P', '!']]

                # inputs = tokenizer.encode_plus(example.text, add_special_tokens=True)
                batch = [tokenizer.encode_plus(ti, add_special_tokens=True, is_pretokenized=True) for ti in tokens]
                max_length += 2
                # batch =   [{'input_ids': [101, 17160, 10124, 169, 102],
                #               'token_type_ids': [0, 0, 0, 0, 0],
                #               'attention_mask': [1, 1, 1, 1, 1]},
                #           {'input_ids': [101, 10124, 169, 16138, 102],
                #               'token_type_ids': [0, 0, 0, 0, 0],
                #               'attention_mask': [1, 1, 1, 1, 1]},
                #           {'input_ids': [101, 169, 16138, 10454, 102],
                #               'token_type_ids': [0, 0, 0, 0, 0],
                #               'attention_mask': [1, 1, 1, 1, 1]},
                #           {'input_ids': [101, 16138, 10454, 11940, 102],
                #               'token_type_ids': [0, 0, 0, 0, 0],
                #               'attention_mask': [1, 1, 1, 1, 1]},
                #           {'input_ids': [101, 10454, 11940, 10142, 102],
                #               'token_type_ids': [0, 0, 0, 0, 0],
                #               'attention_mask': [1, 1, 1, 1, 1]},
                #           {'input_ids': [101, 11940, 10142, 11152, 102],
                #               'token_type_ids': [0, 0, 0, 0, 0],
                #               'attention_mask': [1, 1, 1, 1, 1]},
                #           {'input_ids': [101, 10142, 11152, 81130, 102],
                #               'token_type_ids': [0, 0, 0, 0, 0],
                #               'attention_mask': [1, 1, 1, 1, 1]},
                #           {'input_ids': [101, 11152, 81130, 11127, 102],
                #               'token_type_ids': [0, 0, 0, 0, 0],
                #               'attention_mask': [1, 1, 1, 1, 1]},
                #           {'input_ids': [101, 81130, 11127, 106, 102],
                #               'token_type_ids': [0, 0, 0, 0, 0],
                #               'attention_mask': [1, 1, 1, 1, 1]},
                #           {'input_ids': [101, 11127, 106, 102],
                #               'token_type_ids': [0, 0, 0, 0],
                #               'attention_mask': [1, 1, 1, 1]}]

        except:
            print(example)
        try:
            input_ids, token_type_ids = [inputs["input_ids"] for inputs in batch], [inputs["token_type_ids"] for inputs in
                                                                                    batch]
        except:
            input_ids, token_type_ids = [inputs["input_ids"] for inputs in batch], None
        # input_ids = [[101, 17160, 10124, 169, 102], [101, 16138, 10454, 11940, 102],
        #               [101, 10142, 11152, 81130, 102], [101, 11127, 106, 102]]

        # token_type_ids = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0]]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_masks = [[1 if mask_padding_with_zero else 0] * len(i) for i in input_ids]
        # attention_masks = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1]]


        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids[-1])
        # padding_length = 1

        input_ids[-1] = input_ids[-1] + ([pad_token] * padding_length)
        # input_ids = [[101, 17160, 10124, 169, 102], [101, 16138, 10454, 11940, 102],
        #               [101, 10142, 11152, 81130, 102], [101, 11127, 106, 102, 0]]
        #                                                                       ^

        attention_masks[-1] = attention_masks[-1] + ([0 if mask_padding_with_zero else 1] * padding_length)
        # attention_masks = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
        #                                                                                    ^

        if token_type_ids is not None:
            token_type_ids[-1] = token_type_ids[-1] + ([pad_token_segment_id] * padding_length)
        # token_type_ids = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        # these ate only useful if we have [CLS] text [SEP] text2 [SEP]

        assert int(np.mean([len(t) for t in input_ids])) == max_length, "Error with input length {} vs {}".format(
            np.mean([len(t) for t in input_ids]), max_length)
        assert int(np.mean([len(t) for t in attention_masks])) == max_length, "Error with input length {} vs {}".format(
            np.mean([len(t) for t in attention_masks]), max_length
        )
        if token_type_ids is not None:
            assert int(np.mean([len(t) for t in token_type_ids])) == max_length, "Error with input length {} vs {}".format(
                np.mean([len(t) for t in token_type_ids]), max_length
            )
        assert len(input_ids) <= args.per_gpu_train_batch_size, "Error, there are {} batches".format(len(input_ids))


        # labels = label_map[example.labels]
        try:
            label_ids = example.labels_binary[:]
            # label_ranks = example.label_ranks[:]
        except:
            label_ids = None
            # label_ranks = None

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % input_ids)
            logger.info("attention_mask: %s" % attention_masks)
            logger.info("token_type_ids: %s" % token_type_ids)

            logger.info("label: %r" % label_ids)

        # if len(input_ids) > 10:
        #     print(input_ids[-1])
        # if total_tokens > 2089:
        #     print(sum([bool(i) for i in input_ids[-1]]))
        if label_examples or not doc_batching:
            features.append(
                InputFeatures(
                    input_ids=input_ids[0],
                    input_mask=attention_masks[0],
                    segment_ids=token_type_ids[0] if token_type_ids is not None else None,
                    label_ids=label_ids,
                    # label_ranks=label_ranks,
                    guid=example.guid,
                )
            )
        else:
            # print(label_ids)
            # print(label_ids * len(input_ids))
            # sys.exit()
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=attention_masks,
                    segment_ids=token_type_ids if token_type_ids is not None else None,
                    label_ids=label_ids,
                    # label_ranks=label_ranks,
                    guid=int(example.guid),
                )
            )
            # features.append(
            #     InputFeatures(
            #         input_ids=input_ids,
            #         input_mask=attention_masks,
            #         segment_ids=token_type_ids,
            #         label_ids=[label_ids] * len(input_ids),
            #         label_ranks=[label_ranks] * len(input_ids),
            #         guid=[int(example.guid)] * len(input_ids),
            #     )
            # )
    # print(features)
    # print(len(features))
    return features


def load_and_cache_examples(args, tokenizer, evaluate=False, test=False, label_data=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = MyProcessor(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        f"cached_{'dev' if evaluate else 'test' if test else 'train'}_"
        f"{list(filter(None, args.encoder_name_or_path.split('/'))).pop()}_{str(args.doc_max_seq_length)}_"
        f"{str(args.label_threshold)}_{str(args.doc_batching)}")

    if label_data:
        cached_label_features_file = cached_features_file + '_labels'
    else:
        cached_label_features_file = cached_features_file

    if os.path.exists(cached_label_features_file) and not args.overwrite_cache and not args.preprocess:
        logger.info("Loading features from cached file %s", cached_features_file)
        doc_features = torch.load(cached_features_file)
        if label_data:
            label_features = torch.load(cached_label_features_file)
        idx2id = torch.load(cached_features_file + 'idx2id')
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()

        doc_examples, idx2id = processor.get_dev_examples() if evaluate \
            else processor.get_test_examples() if test else processor.get_train_examples()
        label_examples, junk = processor.get_label_desc()

        # doc_examples = doc_examples[:80]
        doc_features = convert_examples_to_features(
            args,
            doc_examples,
            tokenizer,
            label_list=label_list,
            max_length=args.doc_max_seq_length,
            pad_on_left=bool(args.encoder_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.encoder_type in ["xlnet"] else 0,
            doc_batching=args.doc_batching,
        )
        if label_data:
            label_features = convert_examples_to_features(
                args,
                label_examples,
                tokenizer,
                label_list=label_list,
                max_length=args.label_max_seq_length,
                pad_on_left=bool(args.encoder_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.encoder_type in ["xlnet"] else 0,
                label_examples=True,
            )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(doc_features, cached_features_file)
            if label_data:
                torch.save(label_features, cached_label_features_file)
            torch.save(idx2id, cached_features_file + 'idx2id')

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if args.doc_batching:
        all_input_ids = [torch.tensor(f.input_ids, dtype=torch.long) for f in doc_features]
        all_attention_mask = [torch.tensor(f.input_mask, dtype=torch.long) for f in doc_features]
        all_labels = [torch.tensor(f.label_ids, dtype=torch.long) for f in doc_features]
        all_doc_ids = [torch.tensor(f.guid, dtype=torch.long) for f in doc_features]
        # all_label_ranks = [torch.tensor(f.label_ranks, dtype=torch.long) for f in doc_features]

        # the above all all lists of tensors, each item in the list corresponding to different examples
        # due to the document batching and varying document sizes, these must be in a list and not a large tensor


        if not args.encoder_type == 'xlmroberta':
            all_token_type_ids = [torch.tensor(f.segment_ids, dtype=torch.long) for f in doc_features]
            doc_dataset = BatchedDataset([all_input_ids, all_attention_mask, all_labels,
                                          all_doc_ids, all_token_type_ids])
                                        # all_doc_ids, all_label_ranks, all_token_type_ids])
        else:
            doc_dataset = BatchedDataset([all_input_ids, all_attention_mask, all_labels,
                                          all_doc_ids])
                                          # all_doc_ids, all_label_ranks])
    else:
        all_input_ids = torch.tensor([f.input_ids for f in doc_features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.input_mask for f in doc_features], dtype=torch.long)
        all_labels = torch.tensor([f.label_ids for f in doc_features], dtype=torch.long)
        all_doc_ids = torch.tensor([int(f.guid) for f in doc_features], dtype=torch.long)
        # all_label_ranks = torch.tensor([f.label_ranks for f in doc_features], dtype=torch.long)
        if not args.encoder_type == 'xlmroberta':
            all_token_type_ids = torch.tensor([f.segment_ids for f in doc_features], dtype=torch.long)
            doc_dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels,
                                        all_doc_ids, all_token_type_ids)
                                        # all_doc_ids, all_label_ranks, all_token_type_ids)
        else:
            doc_dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels,
                                        all_doc_ids, all_label_ranks)
                                        # all_doc_ids, all_label_ranks)

    if label_data:
        all_input_ids = torch.tensor([f.input_ids for f in label_features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.input_mask for f in label_features], dtype=torch.long)
        if not args.encoder_type == 'xlmroberta':
            all_token_type_ids = torch.tensor([f.segment_ids for f in label_features], dtype=torch.long)
            label_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
        else:
            label_dataset = TensorDataset(all_input_ids, all_attention_mask)




        return doc_dataset, label_dataset, idx2id
    else:
        return doc_dataset, idx2id


def my_collate(batch):
    input_ids = tuple([b[0] for b in batch])
    attn_mask = tuple([b[1] for b in batch])
    labels = tuple([b[2] for b in batch])
    doc_ids = tuple([b[3] for b in batch])
    label_ranks = tuple([b[4] for b in batch])
    if len(batch[0]) == 5:
        return [input_ids, attn_mask, labels, doc_ids, label_ranks]
    else:
        token_type_ids = tuple([b[5] for b in batch])
        return [input_ids, attn_mask, labels, doc_ids, label_ranks, token_type_ids]


class MyDataParallel(torch.nn.DataParallel):
    """
    Rewrite of torch.nn.DataParallel to support scattering of pre-made, varying-sized batches
    WHY ISN'T THIS ALREADY SUPPORTED???
    """
    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        try:
            kwargs = tuple({'doc_input_ids': kwargs['doc_input_ids'][i].to('cuda:{}'.format(i)),
                            'doc_attention_mask': kwargs['doc_attention_mask'][i].to('cuda:{}'.format(i)),
                            'labels': kwargs['labels'][i].to('cuda:{}'.format(i)),
                            'ranks': kwargs['ranks'][i].to('cuda:{}'.format(i)),
                            'token_type_ids': kwargs['token_type_ids'][i].to('cuda:{}'.format(i)),
                            } for i in range(len(kwargs['doc_input_ids'])))
        except:
            kwargs = tuple({'doc_input_ids': kwargs['doc_input_ids'][i].to('cuda:{}'.format(i)),
                            'doc_attention_mask': kwargs['doc_attention_mask'][i].to('cuda:{}'.format(i)),
                            'labels': kwargs['labels'][i].to('cuda:{}'.format(i)),
                            'ranks': kwargs['ranks'][i].to('cuda:{}'.format(i)),
                            } for i in range(len(kwargs['doc_input_ids'])))

        inputs = tuple(tuple() for i in range(len(kwargs)))



        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)


import threading
import torch
from torch.cuda._utils import _get_device_index
from torch._utils import ExceptionWrapper


def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None

def parallel_apply(modules, inputs, kwargs_tup=None, devices=None):
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.
    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices
    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = list(map(lambda x: _get_device_index(x, True), devices))
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        print(output)
        if isinstance(output, ExceptionWrapper):
            output.reraise()

        outputs.append(output)
    return outputs
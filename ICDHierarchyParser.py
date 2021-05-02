import os
import random
from collections import defaultdict, Counter
import string
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import argparse
import itertools
import numpy as np
from sklearn.preprocessing import normalize
from utils import *
import string
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import json
from torch import save
import copy
import requests
import re
import os
import pickle as pkl

from bs4 import BeautifulSoup
from collections import defaultdict


class ICDHierarchy():
    def __init__(self, args):
        self.args = args
        self.dataset = args.data_dir.split('/')[1]
        self.extant_hierarchy = 'data/hierarchical_data/icd-10-2019-hierarchy.json'
        # from https://github.com/icd-codex/icd-codex/blob/dev/icdcodex/data/icd-10-2019-hierarchy.json
        self.hier_data_path = ''

    def get_dataset_codes(self):
        """
        Loads in all codes from the actual relevant dataset from the data split .tsv files
        :return:
        """
        codes = set()
        for t in ['train', 'dev']:
            with open(os.path.join(self.args.data_dir, '{}_{}_{}.tsv'.format(t, self.args.label_threshold,
                                                                             self.args.ignore_labelless_docs))) as f:
                codes = codes.union(set('|'.join([l.split('\t')[-1] for l in f.read().splitlines()[1:]]).split('|')))
        codes = {c for c in codes if c}
        return codes

    def get_all_codes(self):
        """
        Loads in all codes in their hierarchy from the file containing the codes and their descriptions
        :return:
        """
        with open(os.path.join(self.args.data_dir, 'desc.tsv')) as f:
            codes = set([d.split('\t')[0] for d in f.readlines()])
        return codes

    def build_tree(self):
        """
        Builds a tree from our dataset
        :return:
        """
        all_codes = sorted(self.get_all_codes())
        all_codes = [c for c in all_codes if '-' not in c]
        print("Building ICD10 Hierarchy Tree...")
        for i, code1 in enumerate(tqdm(all_codes)):
            is_parent = True
            while is_parent:
                for code2 in all_codes[i + 1:]:
                    if len(code1) < len(code2) and code1 == code2[:len(code1)]:
                        self.tree[code1].append(code2)
                    else:
                        is_parent = False
                        break
                if i + 1 == len(all_codes) or code2 == all_codes[-1]:
                    break

    def load_extant_hier(self):
        hier = json.load(open(self.extant_hierarchy))['tree']
        return hier

    def build_extant_hier(self):
        hier = self.load_extant_hier()

        def recursively_fill_dict(icd10tree):
            parent = icd10tree['id'].lower()
            if '(' in parent and ')' in parent:
                parent = parent[parent.find("(") + 1:parent.find(")")]
            try:
                children = icd10tree['children']
                for children_dict in children:
                    child = children_dict['id'].lower()
                    if '(' in child and ')' in child:
                        child = child[child.find("(") + 1:child.find(")")]
                    self.external_tree[parent.lower()].append(child)
                    recursively_fill_dict(children_dict)
            except:
                pass

        recursively_fill_dict(hier)

    def fill_final_tree(self, tree):
        for parent, children in tree.items():
            for child in children:
                if self.full_child2parent_dict[child]:
                    parent_already_there = self.full_child2parent_dict[child][0]
                    true_parent = [parent_already_there, parent][np.argmax([len(parent_already_there), len(parent)])]
                    self.full_child2parent_dict[child] = [true_parent]
                else:
                    self.full_child2parent_dict[child].append(parent)
            self.full_parent2children_tree[parent] += children

    def merge_trees(self):
        self.fill_final_tree(self.tree)
        self.fill_final_tree(self.external_tree)

    def parse_icd10(self, parse_own=True):
        self.build_tree()
        self.build_extant_hier()
        self.merge_trees()
        all_codes = set(self.full_child2parent_dict.keys()).union(
            set(i[0] for i in self.full_child2parent_dict.values()))
        all_codes = sorted(all_codes)
        idx2code = {i: code for i, code in enumerate(all_codes)}
        code2idx = {v: k for k, v in idx2code.items()}
        with open(os.path.join(self.hier_data_path, 'icd10hierarchy.txt'), 'w') as f:
            for child, parent in self.full_child2parent_dict.items():
                f.write(str(code2idx[parent[0]]) + ' ' + str(code2idx[child]) + '\n')
        save(idx2code, os.path.join(self.hier_data_path, 'idx2icd10.p'))
        save(code2idx, os.path.join(self.hier_data_path, 'icd102idx.p'))
        try:
            assert self.get_dataset_codes().issubset(set(all_codes))
        except:
            print("The following codes from the dataset are not in the hierarchy:")
            print(set(self.get_dataset_codes()) - set(all_codes))

class SpanishICD10Hierarchy(ICDHierarchy):
    def __init__(self, args):
        super(SpanishICD10Hierarchy, self).__init__(args)
        self.tree = defaultdict(list)  # {parent: [child1, child2...]}
        self.external_tree = defaultdict(list)  # {parent: [child1, child2...]}
        self.full_parent2children_tree = defaultdict(list)  # {parent: [child1, child2...]}
        self.full_child2parent_dict = defaultdict(list)  # {child: [parent1, parent2...]}, so we can fix inconsistencies
        self.hier_data_path = 'data/hierarchical_data/es/'
        if not os.path.exists(self.hier_data_path):
            os.mkdir(self.hier_data_path)
        if not os.path.exists(os.path.join(self.hier_data_path, 'icd10hierarchy.txt')):
            self.parse_icd10()
            save(self.full_parent2children_tree, os.path.join(self.hier_data_path, 'parent2children_tree.p'))
            save(self.full_child2parent_dict, os.path.join(self.hier_data_path, 'child2parent.p'))

class GermanICD10Hierarchy:

    def __init__(self, args):
        self.url = "https://www.dimdi.de/static/de/klassifikationen/icd/icd-10-gm/kode-suche/htmlgm2016/"
        self.build_tree()
        self.link_nodes()
        self.hier_data_path = 'data/hierarchical_data/de/'
        self.args = args
        if not os.path.exists(self.hier_data_path):
            os.mkdir(self.hier_data_path)
        if not os.path.exists(os.path.join(self.hier_data_path, 'icd10hierarchy.txt')):
            self.parse_icd10()

    def build_tree(self):
        rget = requests.get(self.url)
        soup = BeautifulSoup(rget.text, "lxml")
        chapters = soup.findAll("div", {"class": "Chapter"})
        self.tree = dict()
        self.code2title = dict()

        def recurse_chapter_tree(chapter_elem):
            ul = chapter_elem.find("ul")
            codes = {}
            if ul is not None:
                # get direct child only
                ul = ul.find_all(recursive=False)
                for uli in ul:
                    uli_codes = recurse_chapter_tree(uli)
                    codes[uli.a.text] = {
                        # "title": uli.a["title"],
                        "subgroups": uli_codes if uli_codes else None
                    }
                    # self.code2title[uli.a.text] = uli.a["title"]
            return codes

        # used to clean chapter titles
        prefix_re = re.compile(r"Kapitel (?P<chapnum>[IVX]{1,5})")  # I->minlen, XVIII->maxlen

        for chapter in chapters:
            # chapter code and title
            chap_h2 = chapter.find("h2").text[:-9]
            chap_code = chap_h2.strip("()")
            chap_title = prefix_re.sub("", chap_h2)
            chap_num = prefix_re.search(chap_h2).groupdict()['chapnum']
            if chap_num == "XIXV":
                # small fix for "XIXVerletzungen .." V is part of word
                chap_num = "XIX"
            # parse hierarchy
            self.tree[chap_num] = {
                # "title": chap_title,
                # "code": chap_code,
                "subgroups": recurse_chapter_tree(chapter)
            }
            self.code2title[chap_num] = chap_title

    def link_nodes(self):
        self.parent2childs = dict()

        def set_parent2childs(d):
            for k, v in d.items():
                if k not in ("subgroups"):
                    if v["subgroups"] is not None:
                        self.parent2childs[k] = set(v["subgroups"].keys())
                        set_parent2childs(v["subgroups"])

        set_parent2childs(self.tree)

        def update_parent2childs():
            parent2childs = copy.deepcopy(self.parent2childs)

            def get_all_descendants(parent, childs):
                temp_childs = copy.deepcopy(childs)
                for childi in temp_childs:
                    # get child's childs
                    if childi in parent2childs:
                        # recurse till leaf nodes
                        get_all_descendants(childi, parent2childs[childi])
                        parent2childs[parent].update(parent2childs[childi])

            for parent, childs in self.parent2childs.items():
                get_all_descendants(parent, childs)

            self.parent2childs = parent2childs

        update_parent2childs()

        # get reversed mapping
        self.child2parents = defaultdict(set)

        for parent, childs in self.parent2childs.items():
            for childi in childs:
                self.child2parents[childi].add(parent)

    def get_dataset_codes(self):
        """
        Loads in all codes from the actual relevant dataset from the data split .tsv files
        :return:
        """
        codes = set()
        for t in ['train', 'dev']:
            with open(os.path.join(self.args.data_dir, '{}_{}_{}.tsv'.format(t, self.args.label_threshold,
                                                                             self.args.ignore_labelless_docs))) as f:
                codes = codes.union(set('|'.join([l.split('\t')[-1] for l in f.read().splitlines()[1:]]).split('|')))
        codes = {c for c in codes if c}
        return codes

    def rebuild_tree(self):
        hier = self.tree
        tree = defaultdict(list)
        def recursively_fill_dict(icd10tree):
            parents = icd10tree.keys()
            for parent in parents:
                children = icd10tree[parent]['subgroups']
                if children is not None:
                    tree[parent] = list(children.keys())
                    recursively_fill_dict(children)
        recursively_fill_dict(hier)
        return tree

    def parse_icd10(self):
        tree = self.rebuild_tree()
        tree['root'] = list(self.tree.keys())
        save(tree, os.path.join(self.hier_data_path, 'parent2children_tree.p'))
        child2parent_dict = {}
        for parent, children in tree.items():
            for child in children:
                child2parent_dict[child] = parent
        save(child2parent_dict, os.path.join(self.hier_data_path, 'child2parent.p'))

        all_codes = set(tree.keys()).union(set([i for j in tree.values() for i in j]))
        idx2code = {i: code for i, code in enumerate(all_codes)}
        code2idx = {v: k for k, v in idx2code.items()}

        with open(os.path.join(self.hier_data_path, 'icd10hierarchy.txt'), 'w') as f:
            for parent, children in tree.items():
                for child in children:
                    f.write(str(code2idx[parent]) + ' ' + str(code2idx[child]) + '\n')
        save(idx2code, os.path.join(self.hier_data_path, 'idx2icd10.p'))
        save(code2idx, os.path.join(self.hier_data_path, 'icd102idx.p'))
        try:
            assert self.get_dataset_codes().issubset(set(all_codes))
        except:
            print("The following codes from the dataset are not in the hierarchy:")
            print(set(self.get_dataset_codes()) - set(all_codes))

class CantemistICD10Hierarchy(ICDHierarchy):
    def __init__(self, args):
        super(CantemistICD10Hierarchy, self).__init__(args)
        self.tree = defaultdict(list)  # {parent: [child1, child2...]}
        self.external_tree = defaultdict(list)  # {parent: [child1, child2...]}
        self.full_parent2children_tree = defaultdict(list)  # {parent: [child1, child2...]}
        self.full_child2parent_dict = defaultdict(list)  # {child: [parent1, parent2...]}, so we can fix inconsistencies
        self.hier_data_path = 'data/hierarchical_data/cantemist/'
        if not os.path.exists(self.hier_data_path):
            os.mkdir(self.hier_data_path)
        if not os.path.exists(os.path.join(self.hier_data_path, 'icd10hierarchy.txt')):
            self.parse_icd10()
            save(self.full_parent2children_tree, os.path.join(self.hier_data_path, 'parent2children_tree.p'))
            save(self.full_child2parent_dict, os.path.join(self.hier_data_path, 'child2parent.p'))

    def build_tree(self):
        """
        Builds a tree from our dataset
        :return:
        """
        with open('data/hierarchical_data/icd-0-3_shallow_hier.tsv', 'r') as shallow_tree_f:
            dat = shallow_tree_f.read().splitlines()

        for entry in dat:
            try:
                parent, children = entry.split('\t')
                children = children.split(', ')
                self.tree[parent] = children
                for c in children:
                    self.tree[c]
            except:
                self.tree[entry]

        high_level_cats = {(int(s.split('-')[0]), int(s.split('-')[1])) for s in self.tree.keys() if s !='root'}

        all_codes = list(self.get_all_codes())
        all_codes += [c[:6] for c in all_codes]
        all_codes = sorted(set(all_codes))
        print("Building ICD10 Hierarchy Tree...")
        for i, code1 in enumerate(tqdm(all_codes)):
            if len(code1) == 6:
                # we want to find the range (e.g. 935-937) to which it belongs
                code_prefix = int(code1[:3])
                parent_range = [s for s in high_level_cats if s[0]<=code_prefix<=s[1]]
                parent_range = parent_range[np.argmin([s[1] - s[0] for s in parent_range])] if len(parent_range)>1\
                    else parent_range[0]
                parent_range = '-'.join([str(p) for p in parent_range])
                self.tree[parent_range].append(code1)
            is_parent = True
            while is_parent:
                for code2 in all_codes[i + 1:]:
                    if len(code1) < len(code2) and code1 == code2[:len(code1)]:
                        self.tree[code1].append(code2)
                    else:
                        is_parent = False
                        break
                if i + 1 == len(all_codes) or code2 == all_codes[-1]:
                    break

    def parse_icd10(self):
        self.build_tree()
        self.fill_final_tree(self.tree)
        all_codes = set(self.full_child2parent_dict.keys()).union(
            set(i[0] for i in self.full_child2parent_dict.values()))
        all_codes = sorted(all_codes)
        idx2code = {i: code for i, code in enumerate(all_codes)}
        code2idx = {v: k for k, v in idx2code.items()}
        with open(os.path.join(self.hier_data_path, 'icd10hierarchy.txt'), 'w') as f:
            for child, parent in self.full_child2parent_dict.items():
                f.write(str(code2idx[parent[0]]) + ' ' + str(code2idx[child]) + '\n')
        save(idx2code, os.path.join(self.hier_data_path, 'idx2icd10.p'))
        save(code2idx, os.path.join(self.hier_data_path, 'icd102idx.p'))
        try:
            assert self.get_dataset_codes().issubset(set(all_codes))
        except:
            print("The following codes from the dataset are not in the hierarchy:")
            print(set(self.get_dataset_codes()) - set(all_codes))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_threshold",
        default=0,
        type=int,
        help="Exclude labels which occur <= threshold",
    )
    parser.add_argument('--ignore_labelless_docs', action='store_true',
                        help="Whether to ignore documents with no labels.")
    parser.add_argument('--preprocess', action='store_true', help="Whether to redo all of the pre-processing.")
    parser.add_argument('--make_plots', action='store_true', help="Whether to make plots on data.")
    parser.add_argument('--train_on_all', action='store_true')
    parser.add_argument('--data_dir', default='processed_data/cantemist/')
    parser.add_argument('--local_rank', default=-1)
    parser.add_argument('--encoder_name_or_path', type=str, default='xlm-roberta-base')
    parser.add_argument('--doc_max_seq_length', default=256)

    parser.add_argument(
        "--encoder_type",
        default='xlmroberta',
        type=str,
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
    parser.add_argument(
        "--loss_fct", default="none", type=str, help="The function to use.",
    )
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
        "--label_max_seq_length",
        default=15,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--logit_aggregation", type=str, default='max', help="Whether to aggregate logits by max value "
                                                                             "or average value. Options:"
                                                                             "'--max', '--avg'")
    parser.add_argument("--label_attention", action="store_true", help="Whether to use the label attention model")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
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
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument('--do_iterative_class_weights', action='store_true', help="Whether to use iteratively "
                                                                                  "calculated class weights")
    parser.add_argument('--do_normal_class_weights', action='store_true', help="Whether to use normally "
                                                                               "calculated class weights")
    parser.add_argument('--do_ranking_loss', action='store_true', help="Whether to use the ranking loss component.")
    parser.add_argument('--do_weighted_ranking_loss', action='store_true',
                        help="Whether to use the weighted ranking loss component.")
    parser.add_argument('--do_experimental_ranks_instead_of_labels', action='store_true', help='Whether to send ranks '
                                                                                               'instead of binary labels to loss function')
    parser.add_argument('--doc_batching', action='store_true', help="Whether to fit one document into a batch during")

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
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

    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()
    # args.data_dir = 'processed_data/spanish/es/'
    if 'spanish' in args.data_dir:
        gen = SpanishICD10Hierarchy(args)
    elif 'german' in args.data_dir:
        gen = GermanICD10Hierarchy(args)
    elif 'cantemist' in args.data_dir:
        gen = CantemistICD10Hierarchy(args)



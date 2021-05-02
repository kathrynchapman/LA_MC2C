import os
from torch import load
from collections import defaultdict
import argparse
import sys

class HierarchicalEvaluator():
    def __init__(self, args, test=False, reduced=False):
        self.args = args
        self.hier_data_path = 'data/hierarchical_data/'
        if 'spanish' in args.data_dir:
            self.hier_data_path += 'es/'
        elif 'cantemist' in args.data_dir:
            self.hier_data_path += 'cantemist/'
        elif 'german' in args.data_dir:
            self.hier_data_path += 'de/'
        self.code2idx = load(os.path.join(self.hier_data_path, 'icd102idx.p'))
        self.pred_ids = set()
        self.gold_ids = set()
        self.test = test
        self.reduced = reduced

    def load_preds_data(self):
        preds_path = os.path.join(self.args.output_dir, f"preds_{'dev' if not self.test else 'test'}.tsv")
        preds = open(preds_path, 'r').read().splitlines()[1:]
        preds = [d.split('\t') for d in preds]
        return preds

    def generate_hier_data(self, type):
        gold_loader = self.load_gold_data_reduced if self.reduced else self.load_gold_data
        data = gold_loader() if type == 'gold' else self.load_preds_data()
        ids = self.gold_ids if type == 'gold' else self.pred_ids
        hier_data = defaultdict(list)
        for doc_id, code in data:
            ids.add(doc_id)
            if type == 'preds':
                hier_data[doc_id].append(str(self.code2idx[code]))
            else:
                if code:
                    hier_data[doc_id] = [str(self.code2idx[c]) for c in code.split('|')]
        return hier_data

    def load_gold_data(self):
        path2gold = os.path.join(self.args.data_dir, f"{'dev' if not self.test else 'test'}_{self.args.label_threshold}_{self.args.ignore_labelless_docs}.tsv")
        gold = [d.split('\t') for d in open(path2gold, 'r').read().splitlines()[1:]]
        gold = [[d[0], d[2]] for d in gold]

        return gold

    def load_gold_data_reduced(self):
        gold_path = os.path.join(self.args.output_dir, f"gold_{'dev' if not self.test else 'test'}.tsv")
        gold = open(gold_path, 'r').read().splitlines()[1:]
        gold = [d.split('\t') for d in gold]
        return gold

    def write_hier_files(self):
        gold_data = self.generate_hier_data('gold')
        preds_data = self.generate_hier_data('preds')

        data = [preds_data, gold_data]
        types = ['pred', 'gold']

        # ids = list(self.gold_ids.union(self.pred_ids))
        ids = sorted(list(self.gold_ids.intersection(self.pred_ids)))

        out_paths = []

        for hier_data, type in zip(data, types):
            out_file = os.path.join(self.args.output_dir, 'hierarchical_{}.txt'.format(type))
            with open(out_file, 'w') as f:
                for doc_id in ids:
                    f.write(' '.join(hier_data[doc_id]) + '\n')
            out_paths.append(out_file)
        return out_paths



    def do_hierarchical_eval(self):
        preds_path, gold_path = self.write_hier_files()
        # ./ bin / HEMKit cat_hier.txt Golden.txt result.txt 100000 5
        hierarchy_path = os.path.join(self.hier_data_path, 'icd10hierarchy.txt')

        eval_cmd = 'HEMKit/bin/HEMKit {} {} {} {} {}'.format(hierarchy_path, gold_path, preds_path,
                                                             str(self.args.max_hierarchical_distance),
                                                             str(self.args.max_hierarchical_error))
        eval_results = os.popen(eval_cmd).read()
        return eval_results


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
    hierarchical_evaluator = HierarchicalEvaluator(args)
    hier_eval_results = hierarchical_evaluator.do_hierarchical_eval()




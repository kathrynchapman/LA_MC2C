import os, sys
import argparse
from torch import load, save
import glob
import pickle
import copy
import time
from datetime import datetime
from collections import defaultdict
from random import shuffle
from tqdm import tqdm, trange

from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain
from skmultilearn.ensemble import RakelD, RakelO
from skmultilearn.ensemble.partition import LabelSpacePartitioningClassifier
from skmultilearn.ensemble.voting import MajorityVotingClassifier
from skmultilearn.cluster.random import RandomLabelSpaceClusterer
from sklearn.preprocessing import MultiLabelBinarizer
from hierarchical_evaluation import *
from process_data import *
from ICDHierarchyParser import *
import scipy.sparse as sparse
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

from scipy.sparse import lil_matrix, hstack, issparse, coo_matrix
import numpy as np
import sklearn.metrics as metrics
from sklearn.svm import SVC
from scikit_models import *
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from skmultilearn.dataset import load_dataset


class ClassificationPipeline():
    def __init__(self, args):
        self.args = args
        self.mlb = self.load(os.path.join(args.data_dir, 'mlb_0_False.p'))
        self.load_data()
        self.make_output_dir(args.output_dir)

    def load(self, dir):
        """
        Loads a pickled data object/instance
        :param dir: path to the object
        :return: the loaded object
        """
        try:
            return pickle.load(open(dir, 'rb'))
        except:
            return load(dir)

    def save(self, data, dir):
        """
        Pickles and dumps a data object/instance at a specified directory
        :param data: the data to pickle/save
        :param dir: the path (including name) where the object will be saved
        :return:
        """
        try:
            pickle.dump(data, open(dir, 'wb'))
        except:
            save(data, dir)

    def load_data(self):
        """
        Loads in the training, dev, and test data from the command-line specified data directory, and adds them
        as attributes. These are expected to be in the format of a list of tuples:
        list: [(doc_id, text, [0, 1, 0, ..., 0, 1]), ... ]
        :return: None
        """
        self.train = self.load(glob.glob(os.path.join(self.args.data_dir, 'train*.p'))[0])
        self.dev = self.load(glob.glob(os.path.join(self.args.data_dir, 'dev*.p'))[0])
        self.test = self.load(glob.glob(os.path.join(self.args.data_dir, 'test*.p'))[0])

    def timer(self, start, end):
        """
        Computes a runtime based on a provided start and end time
        :param start: float: obtained from time.time()
        :param end: float: obtained from time.time()
        :return: a string representing the duration of a runtime: HH:MM:SS:MS
        """
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

    def process_data(self):
        """
        Processes the data loaded in from self.load_data() s.t. the text is vectorized using TF-IDF
        Then, converts the vectorized X and y data to a scipy.sparse.lil_matrix
        :return: None
        """
        vectorizer = TfidfVectorizer(max_df=.9)
        self.X_train = lil_matrix(vectorizer.fit_transform([d[1] for d in self.train]))
        self.y_train = lil_matrix([d[2] for d in self.train])

        self.X_dev = lil_matrix(vectorizer.transform([d[1] for d in self.dev]))
        self.y_dev = lil_matrix([d[2] for d in self.dev])

        self.X_test = lil_matrix(vectorizer.transform([d[1] for d in self.test]))
        self.y_test = lil_matrix([d[2] for d in self.test])

    def print_stats(self):
        """
        Prints out useful information on the current data sets
        :return: str: an output string which will be written to a .txt file for post-viewing
        """
        output = "************ {} Corpus Stats ************".format(self.args.data_dir.split('/')[1].upper())
        output += "\n# Unique Label Combinations TRAIN: {}".format(np.unique(self.y_train.rows).shape[0])
        output += "\n# Unique Label Combinations DEV: {}".format(np.unique(self.y_dev.rows).shape[0])
        output += "\n# Unique Label Combinations TEST: {}".format(np.unique(self.y_test.rows).shape[0])
        output += "\n# Train Examples: {}".format(self.y_train.shape[0])
        output += "\n# Train Labels: {}".format(self.y_train.shape[1])
        print(output)
        return output

    def write_file(self, dir, out_str):
        """
        Writes a string to a specified output file
        :param dir: the path and file name to write to
        :param out_str: the string to write
        :return:
        """
        with open(dir, 'w') as f:
            f.write(out_str)

    def make_output_dir(self, out_dir):
        """
        Checks if a directory exists; if not, creates it.
        :param out_dir: the directory to check/create
        :return: None
        """
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    def print_header(self, clf_name, base_classifier):
        """
        Prints a "nice" header so we know which classifier the following information is for
        :param clf_name:
        :param base_classifier:
        :return:
        """
        print("****************************************************************************")
        print("*****                     {} + {} model                *****".format(clf_name, base_classifier))
        print("****************************************************************************")

    def generate_SVM_params(self, clf_name):
        parameters = {
            'classifier': [SVC()],
            'classifier__C': uniform(loc=0, scale=5000),
            'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'classifier__degree': [0, 1, 2, 3, 4, 5, 6],
            'classifier__gamma': [2 ** i for i in range(-5, 16)] + ['scale', 'auto'],
            'classifier__shrinking': [True, False],
            'classifier__class_weight': ['balanced', None],
            'classifier__random_state': [0],
        }
        if 'Rakel' in clf_name:
            parameters = {'base_' + k: v for k, v in parameters.items()}
            parameters["base_classifier_require_dense"] = [[True, True]]
            parameters["labelset_size"] = [3, 6, 9, 12, 15, 18, 21]

        return parameters

    def generate_RF_params(self, clf_name):
        parameters = {
            'classifier': [RandomForestClassifier()],
            'classifier__n_estimators': [i for i in range(1, 1001)],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_depth': [i for i in range(1, 100)],
            'classifier__min_samples_split': uniform(loc=0, scale=1),
            'classifier__min_samples_leaf': uniform(loc=0, scale=0.5),
            'classifier__max_features': uniform(loc=0, scale=1),
            'classifier__class_weight': ['balanced', 'balanced_subsample', None],
            'classifier__random_state': [0],

        }
        if 'Rakel' in clf_name:
            parameters = {'base_' + k: v for k, v in parameters.items()}
            parameters["base_classifier_require_dense"] = [[True, True]]
            parameters["labelset_size"] = [3, 6, 9, 12, 15, 18, 21]
        return parameters

    def extract_svm_params_from_best_param_dict(self, best_param_dict, clf_name):
        pref = 'base_classifier__' if 'Rakel' in clf_name else 'classifier__'
        svm_params = {"C": best_param_dict[pref + 'C'],
                      "kernel": best_param_dict[pref + 'kernel'],
                      "gamma": best_param_dict[pref + 'gamma'],
                      "degree": best_param_dict[pref + 'degree'],
                      "class_weight": best_param_dict[pref + 'class_weight'],
                      "shrinking": best_param_dict[pref + 'shrinking'],
                      "random_state": 0}
        return svm_params

    def extract_RF_params_from_best_param_dict(self, best_param_dict, clf_name):
        pref = 'base_classifier__' if 'Rakel' in clf_name else 'classifier__'
        rf_params = {"n_estimators": best_param_dict[pref + 'n_estimators'],
                      "criterion": best_param_dict[pref + 'criterion'],
                      "max_depth": best_param_dict[pref + 'max_depth'],
                      "min_samples_split": best_param_dict[pref + 'min_samples_split'],
                      "min_samples_leaf": best_param_dict[pref + 'min_samples_leaf'],
                      "max_features": best_param_dict[pref + 'max_features'],
                      "class_weight": best_param_dict[pref + 'class_weight'],
                      "random_state": 0,
                      "n_jobs": -1}
        return rf_params

    def parameter_search(self):
        """
        Performs a parameter search for the BinaryRelevance, ClassifierChain, and LabelPowerset models.
        Since Rakel* models take so long to train, we exclude them
        :return: None
        """
        classifiers = {
            "BinaryRelevance": MyBinaryRelevance,
            "ClassifierChain": MyClassifierChain,
            "LabelPowerset": LabelPowerset,
            "RakelD": MyRakelD}
        for clf_name, clf in classifiers.items():
            self.args.output_dir = os.path.join(self.args.output_dir,
                                                '_'.join([clf_name, self.args.base_classifier, 'parameter_search']))
            if not os.path.exists(os.path.join(self.args.output_dir, 'best_args.p')):
                self.print_header(clf_name, self.args.base_classifier)
                self.make_output_dir(self.args.output_dir)
                print("Running grid search..........")
                if self.args.base_classifier == 'svm':
                    parameters = self.generate_SVM_params(clf_name)
                elif self.args.base_classifier == 'randomforest':
                    parameters = self.generate_RF_params(clf_name)

                start_time = time.time()
                clf = RandomizedSearchCV(clf(), parameters, scoring='f1_micro', n_jobs=-1, random_state=0)
                search = clf.fit(self.X_train, self.y_train)
                end_time = time.time()
                best_param_dict = search.best_params_
                out_str = '\n'.join([str(k) + ': ' + str(v) for k, v in best_param_dict.items()])
                out_str += '\nBest F1-Micro:' + str(search.best_score_)
                out_str += "\nParameter search runtime: " + self.timer(start_time, end_time)
                self.write_file(os.path.join(self.args.output_dir, 'best_params.txt'), out_str)
                print(out_str)
                self.save(best_param_dict, os.path.join(self.args.output_dir, 'best_args.p'))
                self.args.output_dir = '/'.join(self.args.output_dir.split('/')[:-1])
            else:
                self.args.output_dir = '/'.join(self.args.output_dir.split('/')[:-1])
                continue

    def run_classification(self):
        """
        Trains and evaluates the BR, CC, LP, and Rakel* models
        Can either load in parameters saved from a parameter search, or use command-line spedified parameters
        :return:
        """
        classifiers = {
            "BinaryRelevance": MyBinaryRelevance,
            "ClassifierChain": MyClassifierChain,
            "LabelPowerset": LabelPowerset,
            "RakelD": MyRakelD,
            "RakelO": MyRakelO
        }

        base_clf = SVC if self.args.base_classifier == 'svm' else RandomForestClassifier

        for clf_name, clf in classifiers.items():
            if self.args.load_best_parameters:
                try:                 
                    best_param_dict = self.load(os.path.join(self.args.output_dir,
                                                             '_'.join(['RakelD' if clf_name=='RakelO' else clf_name,
                                                                                             self.args.base_classifier,
                                                                                             'parameter_search']),
                                                             'best_args.p'))                          
                    params = self.extract_svm_params_from_best_param_dict(
                        best_param_dict, clf_name) if self.args.base_classifier == 'svm' else self.extract_RF_params_from_best_param_dict(
                        best_param_dict, clf_name)

                except:
                    print("Sorry, there are no estimated best parameters for the {} model. Using default or user"
                          "specified instead.".format(clf_name))
                    params = {"C": 2744.068,
                      "kernel": 'sigmoid',
                      "gamma": 0.25,
                      "degree": 0,
                      "class_weight": None,
                      "shrinking": False,
                      "random_state": 0}

            labelset_size = best_param_dict['labelset_size'] if 'Rakel' in clf_name else 3
            # labelset_size = 3

            if 'Rakel' not in clf_name:
                model_args = {"classifier": base_clf(**params), "require_dense": [False, True]}
            else:
                # Rakel* models need different parameters
                model_args = {"base_classifier": base_clf(**params),
                              "base_classifier_require_dense": [True, True],
                              "labelset_size": labelset_size}
                if clf_name == 'RakelO':
                    model_args["model_count"] = 2 * self.y_train.shape[1]

            output_dir_str = '_'.join([str(params[k]) for k in sorted(params.keys())])
            self.args.output_dir = os.path.join(self.args.output_dir, '_'.join([clf_name,
                                                                                self.args.base_classifier,
                                                                                output_dir_str
                                                                                ]))
            self.print_header(clf_name, self.args.base_classifier)
            self.make_output_dir(self.args.output_dir)

            if not os.path.exists(os.path.join(self.args.output_dir, 'model.p')):
                clf = clf(**model_args)
                start_time = time.time()
                now = datetime.now()
                print("Start time: ", now.strftime("%d.%m.%y %H:%M"))
                print("Running training..........")
                clf.fit(self.X_train, self.y_train)
                try:
                    self.save(clf, os.path.join(self.args.output_dir, 'model.p'))
                    print("Trained {} model saved!".format(clf_name))
                except:
                    print("The {} model is too big to save; skipping.".format(clf_name))
                print("End time: ", now.strftime("%d.%m.%y %H:%M"))
                end_time = time.time()
                train_duration = self.timer(start_time, end_time)
            else:  # load in existing model with those parameters if one exists
                clf = self.load(os.path.join(self.args.output_dir, 'model.p'))
                train_duration = None
            self.eval(clf, clf_name, train_duration)

    def eval(self, clf, clf_name, duration=None):
        """
        Evaluates the specified classifier on the command-line specified eval data (dev or test)
        :param clf: the classifier to use for making predictions
        :param duration: str: the string output from self.timer() regarding the training runtime
        :return: None
        """
        print("Running eval..........")
        y_preds = clf.predict(self.X_dev) if self.args.eval_data == 'dev' else clf.predict(self.X_test)
        y_true = self.y_dev if self.args.eval_data == 'dev' else self.y_test

        with open(os.path.join(self.args.output_dir, f"preds_{'test' if self.args.eval_data=='test' else 'dev'}.tsv"), "w") as wf:
            wf.write("file\tcode\n")
            data = self.dev if self.args.eval_data == 'dev' else self.test
            ids = [d[0] for d in data]

            preds = [self.mlb.classes_[y_preds.toarray().astype(int)[i, :].astype(bool)].tolist() for i in range(y_preds.shape[0])]

            id2preds = {val: preds[i] for i, val in enumerate(ids)}
            preds = [id2preds[val] if val in id2preds else [] for i, val in enumerate(ids)]

            for idx, doc_id in enumerate(ids):
                for p in preds[idx]:
                    if p != 'None':
                        line = str(doc_id) + "\t" + p + "\n"
                        wf.write(line)

        n_labels = np.sum(y_preds, axis=1)
        avg_pred_n_labels = np.mean(n_labels)
        avg_true_n_labels = np.mean(np.sum(y_true, axis=1))
        total_uniq = len(np.nonzero(np.sum(y_preds, axis=0))[0])

        out_str = "\n************ {} + {} Performance ************".format(clf_name, self.args.base_classifier)
        if duration:
            out_str += "\nTraining Runtime: {}".format(duration)
        out_str += "\nF1: {}".format(metrics.f1_score(y_true, y_preds, average='micro'))
        out_str += "\nP: {}".format(metrics.precision_score(y_true, y_preds, average='micro'))
        out_str += "\nR: {}".format(metrics.recall_score(y_true, y_preds, average='micro'))
        hierarchical_evaluator = HierarchicalEvaluator(self.args, test=True if self.args.eval_data=='test' else False)
        out_str += "\n--- Hierarchical Metrics ---\n"
        out_str += hierarchical_evaluator.do_hierarchical_eval()
        out_str += "\n--- Additional Info ---"
        out_str += "\nAverage #labels/doc preds: " + str(avg_pred_n_labels)
        out_str += "\nAverage #labels/doc true: " + str(avg_true_n_labels)
        out_str += "\nTotal unique labels predicted: " + str(total_uniq)
        if not os.path.exists(os.path.join(self.args.output_dir, 'eval_results.txt')):
            self.write_file(os.path.join(self.args.output_dir, 'eval_results.txt'), out_str)
        print(out_str)
        self.eval_on_all(testing=True if self.args.eval_data=='test' else False)

    def eval_on_all(self, testing=False):

        def load_gold_data():
            path2gold = os.path.join(self.args.data_dir,
                                     f"{'test' if testing else 'dev'}_{self.args.label_threshold}_{self.args.ignore_labelless_docs}.tsv")
            gold = [d.split('\t') for d in open(path2gold, 'r').read().splitlines()[1:]]
            gold = [[d[0], d[2]] for d in gold]
            return gold

        with open(os.path.join(self.args.output_dir, f"preds_{'test' if testing else 'dev'}.tsv"), 'r') as tf:
            test_preds = tf.read().splitlines()
        test, gold = defaultdict(list), defaultdict(list)
        all_labels = set(self.mlb.classes_)
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
        result = "\nF1: {}".format(metrics.f1_score(gold_labels, test_preds, average='micro'))
        result += "\nP: {}".format(metrics.precision_score(gold_labels, test_preds, average='micro'))
        result += "\nR: {}".format(metrics.recall_score(gold_labels, test_preds, average='micro'))
        print("***** Eval results on All Labels *****")
        print(result)
        self.args.output_dir = '/'.join(self.args.output_dir.split('/')[:-1])


    def main(self):
        """
        Executes the relevant methods based on command-line specifications
        :return:
        """
        self.process_data()
        self.write_file(os.path.join(self.args.output_dir, 'dataset_stats.txt'), self.print_stats())
        if self.args.parameter_search:
            self.parameter_search()
        if self.args.do_train:
            self.run_classification()

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data", default='dev', type=str, help="Whether to evaluate the model on the dev or "
                                                                     "test data.", )
    parser.add_argument("--output_dir", default='scikit_exps_dir/', type=str,
                        help="Where to save the models and results.", )
    parser.add_argument("--featurizer", default='tf-idf', type=str, help="How to represent the data as features.", )
    parser.add_argument("--base_classifier", default='svm', type=str, help="Which base classifier to use: svm, "
                                                                           "randomforest", )
    parser.add_argument("--data_dir", default=None, required=True, type=str, help="Path to directory containing "
                                                                                  "pickled and dumped processed data.")
    parser.add_argument('--parameter_search', action='store_true', help='Whether to perform grid search.')
    parser.add_argument('--do_train', action='store_true', help='Whether to train and evaluate a classifier.')
    parser.add_argument('--load_best_parameters', action='store_true',
                        help='Whether to use the best parameters obtained'
                             'from a parameter search for the model '
                             'training')

    # SVM parameters
    parser.add_argument("--C", default=1.0, type=float, help="Regularization parameter. The strength of the "
                                                             "regularization is inversely proportional to C. Must be "
                                                             "strictly positive. The penalty is a squared l2 penalty.")
    parser.add_argument("--class_weight", default=None, type=str, help="Set the parameter C of class i to "
                                                                       "class_weight[i]*C for SVC. If not given, all "
                                                                       "classes are supposed to have weight one. The "
                                                                       "“balanced” mode uses the values of y to "
                                                                       "automatically adjust weights inversely "
                                                                       "proportional to class frequencies in the input "
                                                                       "data as n_samples / (n_classes * np.bincount(y))")
    parser.add_argument("--kernel", default='rbf', type=str, help="‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’; "
                                                                  "Specifies the kernel type to be used in the "
                                                                  "algorithm. It must be one of ‘linear’, ‘poly’, "
                                                                  "‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. "
                                                                  "If none is given, ‘rbf’ will be used. If a callable "
                                                                  "is given it is used to pre-compute the kernel matrix "
                                                                  "from data matrices; that matrix should be an array "
                                                                  "of shape (n_samples, n_samples).")
    parser.add_argument("--degree", default=3, type=int, help="Degree of the polynomial kernel function (‘poly’). "
                                                              "Ignored by all other kernels.")
    parser.add_argument("--gamma", default='scale', help="Kernel coef. for ‘rbf’, ‘poly’ and ‘sigmoid’.")
    parser.add_argument("--shrinking", default=True, type=bool, help="Whether to use the shrinking heuristic.")

    # Hierarchical Eval Parameters
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

    args = parser.parse_args()
    args.label_threshold = 0
    args.ignore_labelless_docs, args.train_on_all, args.preprocess, args.make_plots = [False] * 4
    args.label_max_seq_length = 15
    args.language = 'cantemist' if 'cantemist' in args.data_dir else ''
    args.language = 'german' if 'german' in args.data_dir else args.language
    args.language = 'spanish' if 'spanish' in args.data_dir else args.language
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.language)

    if 'spanish' in args.data_dir:
        gen = SpanishICD10Hierarchy(args)
    elif 'german' in args.data_dir:
        gen = GermanICD10Hierarchy(args)
    elif 'cantemist' in args.data_dir:
        gen = CantemistICD10Hierarchy(args)

    try:
        pipeline = ClassificationPipeline(args)
    except:
        if 'cantemist' in args.data_dir:
            reader = CantemistReader(args)
        elif 'german' in args.data_dir:
            reader = GermanReader(args)
        elif 'spanish' in args.data_dir:
            reader = SpanishReader(args)
        reader.process_data()
        pipeline = ClassificationPipeline(args)
    pipeline.main()


if __name__ == '__main__':
    main()

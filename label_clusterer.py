import numpy as np
import pickle
from tqdm import tqdm, trange
import math
from collections import defaultdict, Counter, OrderedDict
import operator
import os
import sys
from distance_computer import *


def sort_by_values_len(dict):
    dict_len = {key: len(value) for key, value in dict.items()}

    sorted_key_list = sorted(dict_len.items(), key=operator.itemgetter(1))
    sorted_dict = [tuple([item[0], dict[item[0]]]) for item in sorted_key_list]
    return sorted_dict


class MC2CLabelClusterer():
    def __init__(self, data_dir, max_cluster_size=10, min_cluster_size=5, max_freq_threshold=0.0, add_none=False):
        self.data_dir = data_dir
        self.hierarchical_data_dir = 'data/hierarchical_data/'
        self.hierarchical_data_dir += 'cantemist/' if 'cantemist' in self.data_dir else ''
        self.hierarchical_data_dir += 'es/' if 'spanish' in self.data_dir else ''
        self.hierarchical_data_dir += 'de/' if 'german' in self.data_dir else ''
        self.data_type = ''
        self.load_data('train')
        self.load_data('dev')
        self.add_none = add_none
        self.train_ids = [d[0] for d in self.train_data]
        self.dev_ids = [d[0] for d in self.dev_data]
        self.max_freq_threshold = max_freq_threshold
        self.m = len(self.train_data[0][2])  # num labels
        self.n = len(self.train_data)  # num docs
        self.E = np.zeros((self.n, self.m))
        self.C = np.zeros((self.m, self.m))
        self.create_E_matrix()
        self.power_dict = dict()  # per-label clustering power dictionary
        self.clusters = defaultdict(set)  # dict of {seed: {label1, ..., labeln},...}
        self.loners = set()  # those labels which get their own single binary classifier
        self.doc_id2activated_clusters = defaultdict(set)  # dictionary mapping doc_id --> relevant clusters
        self.max_cluster_size = max_cluster_size
        self.min_cluster_size = min_cluster_size
        self.n_c = (self.m // (self.max_cluster_size - 1)) + 1
        self.cluster_class_counts = dict()
        self.label_class_counts = dict()
        self.overall_idx2cluster_idx = defaultdict(dict)
        self.cluster_idx2overall_idx = defaultdict(dict)
        self.out_dir = os.path.join(self.data_dir,
                                    'MCC/{}_{}_{}_{}'.format(self.min_cluster_size, self.max_cluster_size,
                                                             self.max_freq_threshold, self.add_none))
        self.mlb = pickle.load(open(os.path.join(self.data_dir, 'mlb_0_False.p'), 'rb'))

        if not os.path.exists(os.path.join(self.data_dir, 'MCC/')):
            os.mkdir(os.path.join(self.data_dir, 'MCC/'))
        try:
            self.C = pickle.load(
                open(os.path.join(self.data_dir, 'MCC/C_matrix_{}l_{}d_{}.p'.format(self.m, self.n, self.add_none)),
                     'rb'))
        except:
            self.compute_C_matrix()

        self.delta = self.n_c / self.m
        self.epsilon = 0.001

    def load_data(self, data_type):
        self.train_data = pickle.load(open(os.path.join(self.data_dir, 'train_0_False.p'.format(data_type)), 'rb'))
        self.dev_data = pickle.load(open(os.path.join(self.data_dir, 'dev_0_False.p'.format(data_type)), 'rb'))

    def create_E_matrix(self):
        try:
            for i, (doc_id, text, labels, ranks) in enumerate(self.train_data):
                # labels = np.append(labels, np.array([0]))
                self.E[i, :] = labels
        except:
            for i, (doc_id, text, labels) in enumerate(self.train_data):
                # labels = np.append(labels, np.array([0]))
                self.E[i, :] = labels

    def compute_C_matrix(self):
        """
        Creates a co-occurrence matrix from self.E
        :return:
        """
        print("Computing Co-occurrence Matrix for Label Clustering...")
        for i in trange(self.m):
            for j in range(self.n):
                co_occurrences = set(np.nonzero(self.E[j, :])[0])
                if i not in co_occurrences:
                    continue
                else:
                    for l in co_occurrences:
                        self.C[i, l] = 1
        pickle.dump(self.C, open(os.path.join(self.data_dir, 'MCC/C_matrix_{}l_{}d_{}.p'.format(self.m, self.n,
                                                                                                self.add_none)), 'wb'))

    def generate_clusters_by_freq(self):
        """
        Generates clusters such that the frequencies are as close as possible within the clusters, while maintaining
        mututal exclusivity within each cluster
        :return:
        """
        unordered_freqs = self.compute_label_freqs()  # dict of label: count
        freqs = sorted(unordered_freqs.items(), key=operator.itemgetter(1), reverse=True)  # list of tuples
        # [(code, freq), ...] in descending order of frequency
        max_count = max([d[1] for d in freqs]) if not self.max_freq_threshold else \
            round(self.n * self.max_freq_threshold)  # maximum absolute frequency we'll allow in order for a code
        # to make it in to a cluster

        self.loners = {d[0] for d in freqs if d[1] > max_count}  # set of codes which are too high-freq for a cluster
        freqs = [f for f in freqs if f[0] not in self.loners]  # remove the loners from our frequency list
        remaining_labels = {s[0] for s in freqs}  # set of labels we can still cluster

        non_overlap_dict = {l: np.where(self.C[l, :] == 0)[0].tolist() for l in
                            remaining_labels}  # a dictionary for each
        # label, {label: [list of labels which the key label never co-occurs with], ...}

        for entry in freqs:  # iterate over every label we didn't discard due to high-freq
            seed = entry[0]  # since we're starting with the most high-freq labels, whichever label we're on is the seed
            labels = non_overlap_dict[seed]  # labels which do not overlap with our chosen seed
            if seed not in remaining_labels:  # if we already assigned this label to a cluster, it can't be a seed
                continue
            else:
                labels = [l[0] for l in freqs if l[0] in labels and l[0] in remaining_labels]  # gives us the eligible
                #  labels in descending freq
                cluster = []
                for lab in labels:  # iterate over eligible labels...
                    non_overlaps = set(non_overlap_dict[lab])  # get the set of labels which current potential/eligible
                    # label doesn't co-occur with...
                    if set(cluster).issubset(non_overlaps):  # make sure that none of the labels currently in the
                        # cluster co-occur with this label
                        cluster.append(lab)
                    if len(cluster) == self.max_cluster_size - 1:  # if we reached the maximum cluster size, we're done
                        break

                self.clusters[seed] = cluster + [seed]  # add to dict which keeps track of clusters by their seeds
                remaining_labels -= set(self.clusters[seed])  # remove these clustered labels from our set of remaining
        for l in self.loners:  # make sure to add the loners which we discarded due to high freq; they simply form
            # their own cluster
            self.clusters[l] = [l]

        try:
            temp = self.clusters.deepcopy()
            to_remove = set()
            # now, we want to find those left over, low-frequency codes which ended up alone in their own cluster; the idea
            # is to find a cluster we could add them to, which may cause the max_cluster_size to be exceeded by 1 or 2,
            # but I feel it's better than leaving them on their own, if we want to employ an LDAM loss function
            # sometimes, this doesn't work though and then certain labels are left without a cluster, thus the
            # try/except
            for seed, cluster in self.clusters.items():
                if len(cluster) < self.min_cluster_size and seed not in self.loners:
                    for label in cluster:
                        non_overlaps = set(non_overlap_dict[label])
                        for seed2, cluster2 in self.clusters.items():
                            # if label == seed2 or seed2 in self.loners:
                            if seed == seed2 or seed2 in self.loners:
                                continue
                            else:
                                if set(cluster2).issubset(non_overlaps):
                                    temp[seed2].append(label)
                                    to_remove.add(seed)
                                    break
            for r in to_remove:
                del temp[r]
            assert len(self.mlb.classes_) == len([l for label_list in self.clusters.values() for l in label_list]), \
                "The following labels are not in clusters: {}".format(set(self.mlb.classes_) -
                                                                      set([self.mlb.classes_[l] for label_list in
                                                                           self.clusters.values() for l in label_list]))
            self.clusters = temp
        except:
            pass

        assert len(self.mlb.classes_) == len([l for label_list in self.clusters.values() for l in label_list]), \
            "The following labels are not in clusters: {}".format(set(self.mlb.classes_) -
                                                                  set([self.mlb.classes_[l] for label_list in
                                                                       self.clusters.values() for l in label_list]))

    def compute_label_freqs(self):
        label_freq_dict = {i: np.sum(self.E[:, i]) for i in range(self.m)}
        return label_freq_dict

    def make_cluster_idx2general_idx_dict(self):
        """
        Creates a dictionary which maps between the global index of a label and the local (inner-cluster-specific)
        index. The seed is what identifies a cluster, and this gives us a way to map between what labels are at
        what indices within a cluster
        :return:
        """

        for seed, cluster in self.clusters.items():
            for i, c in enumerate(cluster):
                self.overall_idx2cluster_idx[seed][c] = i  # {Seed: {mlb label idx: inner-cluster idx,...},...}
                self.cluster_idx2overall_idx[seed][i] = c  # {Seed: {inner-cluster idx: mlb label idx,...},...}

    def split_data_by_clusters(self, data_type):
        """
        Splits data such that each cluster (a separate MCC task) has its own dataset
        :return:
        """
        self.data_type = data_type
        self.load_data(data_type)
        if data_type == 'train':
            self.generate_clusters_by_freq()
            data_type = self.train_data
        elif data_type == 'dev':
            data_type = self.dev_data

        mlb = pickle.load(open(os.path.join(self.data_dir, 'mlb_0_False.p'), 'rb'))
        data_dict = defaultdict(list)
        self.make_cluster_idx2general_idx_dict()

        for item in data_type:
            doc_id = item[0]
            text = item[1]
            label_matrix = item[2]
            labels = np.nonzero(label_matrix)[0].tolist()
            activated_clusters = [seed for seed, cluster in self.clusters.items() if bool(set(labels) & set(cluster))]
            if self.data_type == 'train':
                assert len(labels) == len(
                    activated_clusters), "Sorry, mismatch with labels {} and activated clusters {}, " \
                                         "document {}".format('|'.join([str(l) for l in labels]),
                                                              '|'.join([str(a) for a in
                                                                        activated_clusters]),
                                                              doc_id)

            for seed in activated_clusters:
                label_matrix = np.zeros(len(self.clusters[seed]))  # generate an empty matrix of cluster size to add
                # binary labels to for MLCC using BR
                filtered_labels = set(labels) & set(self.clusters[seed])  # determine which labels of this example are
                # relevant to the currently activated cluster
                new_cluster_specific_label_idx = [self.overall_idx2cluster_idx[seed][l] for l in filtered_labels]
                label_matrix[new_cluster_specific_label_idx] = 1
                # if self.data_type == 'train':
                #     print(label_matrix)
                data_dict[seed].append((doc_id, text, label_matrix))

        for seed, data in data_dict.items():
            if not os.path.exists(os.path.join(self.out_dir, str(seed))):
                os.mkdir(os.path.join(self.out_dir, str(seed)))
            with open(os.path.join(self.out_dir, str(seed), self.data_type + '.tsv'), 'w') as f:
                file_name = os.path.join(str(seed), '{}_doc_id2gold.p'.format(self.data_type))
                temp = dict()
                if self.data_type == 'train':
                    all_labs = np.array([e[-1] for e in data])
                    self.label_class_counts[seed] = np.sum(all_labs, axis=0).tolist()
                for entry in data:
                    # if self.data_type == 'train':
                    #     lab_idx = np.nonzero(entry[-1])[0].item()
                    #     self.label_class_counts[seed][lab_idx] += 1  # for class counts

                    self.doc_id2activated_clusters[entry[0]].add(seed)
                    temp[entry[0]] = entry[2]
                    lab = self.cluster_idx2overall_idx[seed][np.nonzero(entry[2])[0][0]]
                    lab = mlb.classes_[lab]
                    f.write(str(entry[0]) + '\t' + entry[1] + '\t' + lab + '\n')

                pickle.dump(temp, open(os.path.join(self.out_dir, file_name), 'wb'))
                pickle.dump(self.overall_idx2cluster_idx[seed],
                            open(os.path.join(self.out_dir, str(seed), 'overall_idx2cluster_idx.p'), 'wb'))
                pickle.dump(self.cluster_idx2overall_idx[seed],
                            open(os.path.join(self.out_dir, str(seed), 'cluster_idx2overall_idx.p'), 'wb'))

    def split_into_cluster_prediction_data(self):
        self.seed2cluster_idx = {c: i for i, c in enumerate(self.clusters.keys())}
        self.cluster_idx2seed = {i: c for i, c in enumerate(self.clusters.keys())}
        temp = []

        with open(os.path.join(self.out_dir, 'doc_ids2clusters.p'), 'wb') as pickle_f, \
                open(os.path.join(self.out_dir, 'doc_ids2clusters.tsv'), 'w') as f:
            for doc_id in self.train_ids:
                activated_clusters = self.doc_id2activated_clusters[doc_id]
                temp.append(len(activated_clusters))
                vectorized_activated_clusters = np.zeros(len(self.clusters))
                vectorized_activated_clusters[[self.seed2cluster_idx[i] for i in activated_clusters]] = 1
                self.doc_id2activated_clusters[doc_id] = vectorized_activated_clusters
                f.write(doc_id + '\t' + '|'.join([str(a) for a in activated_clusters]) + '\n')
            for doc_id in self.dev_ids:
                activated_clusters = self.doc_id2activated_clusters[doc_id]
                temp.append(len(activated_clusters))
                vectorized_activated_clusters = np.zeros(len(self.clusters))
                vectorized_activated_clusters[[self.seed2cluster_idx[i] for i in activated_clusters]] = 1
                self.doc_id2activated_clusters[doc_id] = vectorized_activated_clusters
                f.write(str(doc_id) + '\t' + '|'.join([str(a) for a in activated_clusters]) + '\n')
            pickle.dump(self.doc_id2activated_clusters, pickle_f)

        pickle.dump(self.seed2cluster_idx, open(os.path.join(self.out_dir, 'seed2cluster_idx.p'), 'wb'))
        pickle.dump(self.cluster_idx2seed, open(os.path.join(self.out_dir, 'cluster_idx2seed.p'), 'wb'))
        print("Average number of activated clusters per input doc:", np.mean(temp))
        print("Total number of clusters:", len(self.clusters))

    def generate_preliminary_exp_data(self):
        PRELIMINARY_EXP_DATA = []
        with open(os.path.join(self.out_dir, 'train.p'), 'wb') as pf:
            for d in self.train_data:
                PRELIMINARY_EXP_DATA.append((d[0], d[1], self.doc_id2activated_clusters[d[0]]))
            pickle.dump(PRELIMINARY_EXP_DATA, pf)

    def compute_class_counts(self):

        # label_class_counts = dict()
        # for seed, idx2count_dict in self.label_class_counts.items():
        #     label_class_counts[seed] = [idx2count_dict[k] for k in sorted(idx2count_dict)]
        # print(label_class_counts)

        cluster_class_counts = {seed: sum(counts) for seed, counts in self.label_class_counts.items()}

        cluster_class_counts = [cluster_class_counts[self.cluster_idx2seed[i]] for i in
                                range(len(cluster_class_counts))]

        cluster_class_counts = [(self.n - c) / c for c in cluster_class_counts]
        label_class_counts = {self.seed2cluster_idx[k]: v for k, v in self.label_class_counts.items()}
        # the inner-cluster class counts are now in a dictionary where they are looked up by their index in the
        # MLCC step, rather then by their seed
        pickle.dump(label_class_counts, open(os.path.join(self.out_dir, 'local_class_counts.p'), 'wb'))
        pickle.dump(cluster_class_counts, open(os.path.join(self.out_dir, 'global_cluster_counts.p'), 'wb'))

    def generate_activated_clusters_for_dev_data(self):
        print(self.clusters)
        pass

    def main(self):
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
            self.split_data_by_clusters('train')
            self.split_data_by_clusters('dev')
            self.split_into_cluster_prediction_data()
            self.generate_preliminary_exp_data()
            self.compute_class_counts()
            pickle.dump(self.clusters, open(os.path.join(self.out_dir, 'clusters.p'), 'wb'))
        else:
            self.clusters = pickle.load(open(os.path.join(self.out_dir, 'clusters.p'), 'rb'))


class MC2CLabelClusterer_None(MC2CLabelClusterer):
    def __init__(self, data_dir, max_cluster_size=10, min_cluster_size=5, max_freq_threshold=0.0, add_none=True):
        MC2CLabelClusterer.__init__(self, data_dir, max_cluster_size=max_cluster_size,
                                    min_cluster_size=min_cluster_size,
                                    max_freq_threshold=max_freq_threshold, add_none=add_none)
        try:
            self.C = pickle.load(
                open(os.path.join(self.data_dir, 'MCC/C_matrix_{}l_{}d_{}.p'.format(self.m, self.n, self.add_none)),
                     'rb'))
        except:
            self.compute_C_matrix()

    def generate_clusters_by_freq(self):
        """
        Generates clusters such that the frequencies are as close as possible within the clusters, while maintaining
        mututal exclusivity within each cluster
        :return:
        """
        unordered_freqs = self.compute_label_freqs()  # dict of label: count
        freqs = sorted(unordered_freqs.items(), key=operator.itemgetter(1), reverse=True)  # list of tuples
        # [(code, freq), ...] in descending order of frequency
        max_count = max([d[1] for d in freqs]) if not self.max_freq_threshold else \
            round(self.n * self.max_freq_threshold)  # maximum absolute frequency we'll allow in order for a code
        # to make it in to a cluster
        self.loners = {d[0] for d in freqs if d[1] > max_count}  # set of codes which are too high-freq for a cluster

        freqs = [f for f in freqs if f[0] not in self.loners]  # remove the loners from our frequency list
        remaining_labels = {s[0] for s in freqs}  # set of labels we can still cluster

        non_overlap_dict = {l: np.where(self.C[l, :] == 0)[0].tolist() for l in
                            remaining_labels}  # a dictionary for each
        # label, {label: [list of labels which the key label never co-occurs with], ...}

        non_overlap_dict[self.m] = [f[0] for f in freqs]  # None label co-occurs with nothing

        for entry in freqs:  # iterate over every label we didn't discard due to high-freq
            seed = entry[0]  # since we're starting with the most high-freq labels, whichever label we're on is the seed
            labels = non_overlap_dict[seed]  # labels which do not overlap with our chosen seed
            if seed not in remaining_labels:  # if we already assigned this label to a cluster, it can't be a seed
                continue
            else:
                labels = [l[0] for l in freqs if l[0] in labels and l[0] in remaining_labels]  # gives us the eligible
                #  labels in descending freq
                cluster = []
                for lab in labels:  # iterate over eligible labels...
                    non_overlaps = set(non_overlap_dict[lab])  # get the set of labels which current potential/eligible
                    # label doesn't co-occur with...
                    if set(cluster).issubset(non_overlaps):  # make sure that none of the labels currently in the
                        # cluster co-occur with this label
                        cluster.append(lab)
                    if len(cluster) == self.max_cluster_size - 1:  # if we reached the maximum cluster size, we're done
                        break
                cluster = cluster + [self.m]

                # cluster = [self.m] + cluster

                self.clusters[seed] = [seed] + cluster  # add to dict which keeps track of clusters by their seeds
                remaining_labels -= set(self.clusters[seed])  # remove these clustered labels from our set of remaining
        for l in self.loners:  # make sure to add the loners which we discarded due to high freq; they simply form
            # their own cluster
            # self.clusters[l] = [self.m, l]
            self.clusters[l] = [l, self.m]

        try:
            temp = self.clusters.deepcopy()
            to_remove = set()
            # now, we want to find those left over, low-frequency codes which ended up alone in their own cluster; the idea
            # is to find a cluster we could add them to, which may cause the max_cluster_size to be exceeded by 1 or 2,
            # but I feel it's better than leaving them on their own, if we want to employ an LDAM loss function
            # sometimes, this doesn't work though and then certain labels are left without a cluster, thus the
            # try/except
            for seed, cluster in self.clusters.items():
                if len(cluster) < self.min_cluster_size and seed not in self.loners:
                    for label in cluster:
                        non_overlaps = set(non_overlap_dict[label])
                        for seed2, cluster2 in self.clusters.items():
                            # if label == seed2 or seed2 in self.loners:
                            if seed == seed2 or seed2 in self.loners:
                                continue
                            else:
                                if set(cluster2).issubset(non_overlaps):
                                    temp[seed2].append(label)
                                    to_remove.add(seed)
                                    break
            for r in to_remove:
                del temp[r]
            assert len(self.mlb.classes_)+1 == len({l for label_list in self.clusters.values() for l in label_list}), \
                "The following labels are not in clusters: {}".format(set(self.mlb.classes_) -
                                                                      set([self.mlb.classes_[l] for label_list in
                                                                           self.clusters.values() for l in label_list]))
            self.clusters = temp
        except:
            pass

        assert len(self.mlb.classes_)+1 == len({l for label_list in self.clusters.values() for l in label_list}), \
            "The following labels are not in clusters: {}".format(set(self.mlb.classes_) -
                                                                  set([self.mlb.classes_[l] for label_list in
                                                                       self.clusters.values() for l in label_list]))

    def split_data_by_clusters(self, data_type):
        """
        Splits data such that each cluster (a separate MCC task) has its own dataset
        :return:
        """
        self.data_type = data_type
        self.load_data(data_type)
        if data_type == 'train':
            self.generate_clusters_by_freq()
            data_type = self.train_data
        elif data_type == 'dev':
            data_type = self.dev_data

        mlb = pickle.load(open(os.path.join(self.data_dir, 'mlb_0_False.p'), 'rb'))
        mlb.classes_ = np.append(mlb.classes_, np.array(['None']))
        data_dict = defaultdict(list)
        self.make_cluster_idx2general_idx_dict()

        for item in data_type:
            doc_id = item[0]
            text = item[1]
            label_matrix = item[2]  # e.g. if we have only 10 labels total [0, 1, 0, 0, 0, 1, 1, 0, 0, 0]
            labels = np.nonzero(label_matrix)[0].tolist()  # [1, 5, 6]

            for seed in self.clusters.keys():  # for cluster in clusters
                label_matrix = np.zeros(len(self.clusters[seed]))  # generate an empty matrix of cluster size to add
                # binary labels to for MCC
                filtered_labels = set(labels) & set(self.clusters[seed])  # determine which labels of this example are
                # relevant to the currently activated cluster
                if not filtered_labels:  # if there is no true label in the activate cluster, make it the 'None' label
                    filtered_labels = {self.m}

                new_cluster_specific_label_idx = [self.overall_idx2cluster_idx[seed][l] for l in filtered_labels]
                # now we mapped from the pos labels from the entire label space, to those which are positive
                # within the specific cluster we're focusing on; here, most of the pos labels will be in the -1
                # index, since we made 'None' -1 everywhere (in each cluster, in mbl.classes_...)
                label_matrix[new_cluster_specific_label_idx] = 1

                if data_type == self.train_data:
                    assert np.sum(label_matrix) == 1, "There is more than one active label in the cluster."
                data_dict[seed].append((doc_id, text, label_matrix))

        """
        Okay anything from here down I gotta redo; I need to make sure that when 'None' is the true label in a
        cluster, that the cluster activator has a 0 for that cluster.
        """
        for seed, data in data_dict.items():
            # for cluster, [(doc_id, text, label_matrix_for_that_cluster), (doc_id, text, label_matrix_for_that_cluster)]
            if not os.path.exists(os.path.join(self.out_dir, str(seed))):
                os.mkdir(os.path.join(self.out_dir, str(seed)))
            with open(os.path.join(self.out_dir, str(seed), self.data_type + '.tsv'), 'w') as f:
                file_name = os.path.join(str(seed), '{}_doc_id2gold.p'.format(self.data_type))
                temp = dict()
                if self.data_type == 'train':
                    all_labs = np.array([e[-1] for e in data])
                    self.label_class_counts[seed] = np.sum(all_labs, axis=0).tolist()
                # print(seed, all_labs_counts)
                for entry in data:  # iterate over the different (doc_id, text, label_matrix_for_that_cluster)
                    if entry[-1][-1] != 1:  # make sure 'None' isn't the pos label
                        self.doc_id2activated_clusters[entry[0]].add(seed)
                    temp[entry[0]] = entry[2]  # {doc_id:  label_matrix_for_that_cluster}
                    lab = self.cluster_idx2overall_idx[seed][np.nonzero(entry[2])[0][0]]
                    lab = mlb.classes_[lab]
                    f.write(str(entry[0]) + '\t' + str(entry[1]) + '\t' + str(lab) + '\n')

                pickle.dump(temp, open(os.path.join(self.out_dir, file_name), 'wb'))
                pickle.dump(self.overall_idx2cluster_idx[seed],
                            open(os.path.join(self.out_dir, str(seed), 'overall_idx2cluster_idx.p'), 'wb'))
                pickle.dump(self.cluster_idx2overall_idx[seed],
                            open(os.path.join(self.out_dir, str(seed), 'cluster_idx2overall_idx.p'), 'wb'))

    def split_into_cluster_prediction_data(self):
        self.seed2cluster_idx = {c: i for i, c in enumerate(self.clusters.keys())}
        self.cluster_idx2seed = {i: c for i, c in enumerate(self.clusters.keys())}
        temp = []
        with open(os.path.join(self.out_dir, 'doc_ids2clusters.p'), 'wb') as pickle_f, \
                open(os.path.join(self.out_dir, 'doc_ids2clusters.tsv'), 'w') as f:
            for doc_id in self.train_ids:
                true_pos_clusters = self.doc_id2activated_clusters[doc_id]
                temp.append(len(true_pos_clusters))
                vectorized_activated_clusters = np.zeros(len(self.clusters))  # make empty vector of length n_clusters
                vectorized_activated_clusters[[self.seed2cluster_idx[i] for i in true_pos_clusters]] = 1

                self.doc_id2activated_clusters[doc_id] = vectorized_activated_clusters
                f.write(doc_id + '\t' + '|'.join([str(a) for a in true_pos_clusters]) + '\n')
            for doc_id in self.dev_ids:
                true_pos_clusters = self.doc_id2activated_clusters[doc_id]
                temp.append(len(true_pos_clusters))
                vectorized_activated_clusters = np.zeros(len(self.clusters))
                vectorized_activated_clusters[[self.seed2cluster_idx[i] for i in true_pos_clusters]] = 1
                self.doc_id2activated_clusters[doc_id] = vectorized_activated_clusters
                f.write(str(doc_id) + '\t' + '|'.join([str(a) for a in true_pos_clusters]) + '\n')
            pickle.dump(self.doc_id2activated_clusters, pickle_f)

        pickle.dump(self.seed2cluster_idx, open(os.path.join(self.out_dir, 'seed2cluster_idx.p'), 'wb'))
        pickle.dump(self.cluster_idx2seed, open(os.path.join(self.out_dir, 'cluster_idx2seed.p'), 'wb'))
        print("Average number of activated clusters per input doc:", np.mean(temp))
        print("Total number of clusters:", len(self.clusters))

    def compute_class_counts(self):

        cluster_class_counts = {seed: sum(counts[:-1]) for seed, counts in self.label_class_counts.items()}
        cluster_class_counts = [cluster_class_counts[self.cluster_idx2seed[i]] for i in
                                range(len(cluster_class_counts))]

        cluster_class_counts = [(self.n - c) / c for c in cluster_class_counts]
        label_class_counts = {self.seed2cluster_idx[k]: v for k, v in self.label_class_counts.items()}
        # the inner-cluster class counts are now in a dictionary where they are looked up by their index in the
        # MLCC step, rather then by their seed

        pickle.dump(label_class_counts, open(os.path.join(self.out_dir, 'local_class_counts.p'), 'wb'))
        pickle.dump(cluster_class_counts, open(os.path.join(self.out_dir, 'global_cluster_counts.p'), 'wb'))


class MC2CHierarchicalLabelClusterer_None(MC2CLabelClusterer_None):
    def __init__(self, data_dir, max_cluster_size=10, min_cluster_size=5, max_freq_threshold=0.0, add_none=False):
        super(MC2CHierarchicalLabelClusterer_None, self).__init__(data_dir, max_cluster_size=max_cluster_size,
                                                             min_cluster_size=min_cluster_size,
                                                             max_freq_threshold=max_freq_threshold,
                                                             add_none=add_none)
        self.out_dir = os.path.join(self.data_dir,
                                    'MCC/{}_{}_{}_{}_Hierarchical_Clustering'.format(self.min_cluster_size,
                                                                                     self.max_cluster_size,
                                                                                     self.max_freq_threshold,
                                                                                     self.add_none))
        try:
            self.H = pickle.load(
                open(os.path.join(self.data_dir, 'MCC/H_matrix_{}l_{}d_{}.p'.format(self.m, self.n, self.add_none)),
                     'rb'))
            self.mlb = pickle.load(open(os.path.join(self.data_dir, 'mlb_0_False.p'), 'rb'))
            self.class2idx = {cls: i for i, cls in enumerate(self.mlb.classes_)}

        except:
            self.compute_hierarchical_distances()
            self.class2idx = {cls: i for i, cls in enumerate(self.mlb.classes_)}

    def compute_hierarchical_distances(self):
        self.H = np.zeros((self.m, self.m))
        self.mlb = pickle.load(open(os.path.join(self.data_dir, 'mlb_0_False.p'), 'rb'))
        all_classes = self.mlb.classes_
        dc = DistanceComputer(self.hierarchical_data_dir)

        print("Computing pairwise hierarchical distances between the labels...")
        for i, class1 in enumerate(tqdm(all_classes)):
            remaining_classes = all_classes[i + 1:]
            for j, class2 in enumerate(remaining_classes):
                class1_idx, class2_idx = i, j + i + 1
                distance = dc.compute_distance(class1, class2)
                self.H[class1_idx, class2_idx] = distance
                self.H[class2_idx, class1_idx] = distance
        pickle.dump(self.H, open(os.path.join(self.data_dir, 'MCC/H_matrix_{}l_{}d_{}.p'.format(self.m, self.n,
                                                                                                self.add_none)), 'wb'))

    def generate_clusters_by_hierarchical_distance(self):
        """
        Generates clusters such that the frequencies are as close as possible within the clusters, while maintaining
        mututal exclusivity within each cluster
        :return:
        """
        unordered_freqs = self.compute_label_freqs()  # dict of label: count
        freqs = sorted(unordered_freqs.items(), key=operator.itemgetter(1), reverse=True)  # list of tuples
        # [(code, freq), ...] in descending order of frequency
        max_count = max([d[1] for d in freqs]) if not self.max_freq_threshold else \
            round(self.n * self.max_freq_threshold)  # maximum absolute frequency we'll allow in order for a code
        # to make it in to a cluster
        self.loners = {d[0] for d in freqs if d[1] > max_count}  # set of codes which are too high-freq for a cluster

        freqs = [f for f in freqs if f[0] not in self.loners]  # remove the loners from our frequency list
        remaining_labels = {s[0] for s in freqs}  # set of labels we can still cluster

        non_overlap_dict = {l: np.where(self.C[l, :] == 0)[0].tolist() for l in
                            remaining_labels}  # a dictionary for each
        # label, {label: [list of labels which the key label never co-occurs with], ...}

        non_overlap_dict[self.m] = [f[0] for f in freqs]  # None label co-occurs with nothing

        print("Generating hierarchical clusters...")
        for entry in tqdm(freqs):  # iterate over every label we didn't discard due to high-freq
            seed = entry[0]  # since we're starting with the most high-freq labels, whichever label we're on is the seed
            labels = non_overlap_dict[seed]  # labels which do not overlap with our chosen seed
            if seed not in remaining_labels:  # if we already assigned this label to a cluster, it can't be a seed
                continue
            else:
                sorted_closeby_labels = np.argsort(self.H[seed, :])
                labels = [l for l in sorted_closeby_labels if l in labels and l in remaining_labels]  # gives us
                # elegible labels in ascending distance from our seed

                cluster = []
                for lab in labels:  # iterate over eligible labels...
                    non_overlaps = set(non_overlap_dict[lab])  # get the set of labels which current potential/eligible
                    # label doesn't co-occur with...
                    if set(cluster).issubset(non_overlaps):  # make sure that none of the labels currently in the
                        # cluster co-occur with this label
                        cluster.append(lab)
                    if len(cluster) == self.max_cluster_size - 1:  # if we reached the maximum cluster size, we're done
                        break
                cluster = cluster + [self.m]

                # cluster = [self.m] + cluster

                self.clusters[seed] = [seed] + cluster  # add to dict which keeps track of clusters by their seeds
                remaining_labels -= set(self.clusters[seed])  # remove these clustered labels from our set of remaining
        for l in self.loners:  # make sure to add the loners which we discarded due to high freq; they simply form
            # their own cluster
            # self.clusters[l] = [self.m, l]
            self.clusters[l] = [l, self.m]

        try:
            temp = self.clusters.deepcopy()
            to_remove = set()
            # now, we want to find those left over, low-frequency codes which ended up alone in their own cluster; the idea
            # is to find a cluster we could add them to, which may cause the max_cluster_size to be exceeded by 1 or 2,
            # but I feel it's better than leaving them on their own, if we want to employ an LDAM loss function
            # sometimes, this doesn't work though and then certain labels are left without a cluster, thus the
            # try/except
            for seed, cluster in self.clusters.items():
                if len(cluster) < self.min_cluster_size and seed not in self.loners:
                    for label in cluster:
                        non_overlaps = set(non_overlap_dict[label])
                        for seed2, cluster2 in self.clusters.items():
                            # if label == seed2 or seed2 in self.loners:
                            if seed == seed2 or seed2 in self.loners:
                                continue
                            else:
                                if set(cluster2).issubset(non_overlaps):
                                    temp[seed2].append(label)
                                    to_remove.add(seed)
                                    break
            for r in to_remove:
                del temp[r]
            assert len(self.mlb.classes_)+1 == len({l for label_list in self.clusters.values() for l in label_list}), \
                "The following labels are not in clusters: {}".format(set(self.mlb.classes_) -
                                                                      set([self.mlb.classes_[l] for label_list in
                                                                           self.clusters.values() for l in label_list]))
            self.clusters = temp
        except:
            pass

        assert len(self.mlb.classes_)+1 == len({l for label_list in self.clusters.values() for l in label_list}), \
            "The following labels are not in clusters: {}".format(set(self.mlb.classes_) -
                                                                  set([self.mlb.classes_[l] for label_list in
                                                                       self.clusters.values() for l in label_list]))

    def split_data_by_clusters(self, data_type):
        """
        Splits data such that each cluster (a separate MCC task) has its own dataset
        :return:
        """
        self.data_type = data_type
        self.load_data(data_type)
        if data_type == 'train':
            self.generate_clusters_by_hierarchical_distance()
            data_type = self.train_data
        elif data_type == 'dev':
            data_type = self.dev_data

        mlb = pickle.load(open(os.path.join(self.data_dir, 'mlb_0_False.p'), 'rb'))
        mlb.classes_ = np.append(mlb.classes_, np.array(['None']))
        data_dict = defaultdict(list)
        self.make_cluster_idx2general_idx_dict()

        for item in data_type:
            doc_id = item[0]
            text = item[1]
            label_matrix = item[2]  # e.g. if we have only 10 labels total [0, 1, 0, 0, 0, 1, 1, 0, 0, 0]
            labels = np.nonzero(label_matrix)[0].tolist()  # [1, 5, 6]

            for seed in self.clusters.keys():  # for cluster in clusters
                label_matrix = np.zeros(len(self.clusters[seed]))  # generate an empty matrix of cluster size to add
                # binary labels to for MCC
                filtered_labels = set(labels) & set(self.clusters[seed])  # determine which labels of this example are
                # relevant to the currently activated cluster
                if not filtered_labels:  # if there is no true label in the activate cluster, make it the 'None' label
                    filtered_labels = {self.m}

                new_cluster_specific_label_idx = [self.overall_idx2cluster_idx[seed][l] for l in filtered_labels]
                # now we mapped from the pos labels from the entire label space, to those which are positive
                # within the specific cluster we're focusing on; here, most of the pos labels will be in the -1
                # index, since we made 'None' -1 everywhere (in each cluster, in mbl.classes_...)
                label_matrix[new_cluster_specific_label_idx] = 1

                if data_type == self.train_data:
                    assert np.sum(label_matrix) == 1, "There is more than one active label in the cluster."
                data_dict[seed].append((doc_id, text, label_matrix))

        """
        Okay anything from here down I gotta redo; I need to make sure that when 'None' is the true label in a
        cluster, that the cluster activator has a 0 for that cluster.
        """
        for seed, data in data_dict.items():
            # for cluster, [(doc_id, text, label_matrix_for_that_cluster), (doc_id, text, label_matrix_for_that_cluster)]
            if not os.path.exists(os.path.join(self.out_dir, str(seed))):
                os.mkdir(os.path.join(self.out_dir, str(seed)))
            with open(os.path.join(self.out_dir, str(seed), self.data_type + '.tsv'), 'w') as f:
                file_name = os.path.join(str(seed), '{}_doc_id2gold.p'.format(self.data_type))
                temp = dict()
                if self.data_type == 'train':
                    all_labs = np.array([e[-1] for e in data])
                    self.label_class_counts[seed] = np.sum(all_labs, axis=0).tolist()
                # print(seed, all_labs_counts)
                for entry in data:  # iterate over the different (doc_id, text, label_matrix_for_that_cluster)
                    if entry[-1][-1] != 1:  # make sure 'None' isn't the pos label
                        self.doc_id2activated_clusters[entry[0]].add(seed)
                    temp[entry[0]] = entry[2]  # {doc_id:  label_matrix_for_that_cluster}
                    lab = self.cluster_idx2overall_idx[seed][np.nonzero(entry[2])[0][0]]
                    lab = mlb.classes_[lab]
                    f.write(str(entry[0]) + '\t' + str(entry[1]) + '\t' + str(lab) + '\n')

                pickle.dump(temp, open(os.path.join(self.out_dir, file_name), 'wb'))
                pickle.dump(self.overall_idx2cluster_idx[seed],
                            open(os.path.join(self.out_dir, str(seed), 'overall_idx2cluster_idx.p'), 'wb'))
                pickle.dump(self.cluster_idx2overall_idx[seed],
                            open(os.path.join(self.out_dir, str(seed), 'cluster_idx2overall_idx.p'), 'wb'))


class MC2CHierarchicalLabelClusterer(MC2CLabelClusterer):
    def __init__(self, data_dir, max_cluster_size=10, min_cluster_size=5, max_freq_threshold=0.0, add_none=False):
        super(MC2CHierarchicalLabelClusterer, self).__init__(data_dir, max_cluster_size=max_cluster_size,
                                                             min_cluster_size=min_cluster_size,
                                                             max_freq_threshold=max_freq_threshold,
                                                             add_none=add_none)
        self.out_dir = os.path.join(self.data_dir,
                                    'MCC/{}_{}_{}_{}_Hierarchical_Clustering'.format(self.min_cluster_size,
                                                                                     self.max_cluster_size,
                                                                                     self.max_freq_threshold,
                                                                                     self.add_none))
        try:
            self.H = pickle.load(
                open(os.path.join(self.data_dir, 'MCC/H_matrix_{}l_{}d_{}.p'.format(self.m, self.n, self.add_none)),
                     'rb'))
            self.mlb = pickle.load(open(os.path.join(self.data_dir, 'mlb_0_False.p'), 'rb'))
            self.class2idx = {cls: i for i, cls in enumerate(self.mlb.classes_)}

        except:
            self.compute_hierarchical_distances()
            self.class2idx = {cls: i for i, cls in enumerate(self.mlb.classes_)}

    def compute_hierarchical_distances(self):
        self.H = np.zeros((self.m, self.m))
        self.mlb = pickle.load(open(os.path.join(self.data_dir, 'mlb_0_False.p'), 'rb'))
        all_classes = self.mlb.classes_
        dc = DistanceComputer(self.hierarchical_data_dir)

        print("Computing pairwise hierarchical distances between the labels...")
        for i, class1 in enumerate(tqdm(all_classes)):
            remaining_classes = all_classes[i + 1:]
            for j, class2 in enumerate(remaining_classes):
                class1_idx, class2_idx = i, j + i + 1
                distance = dc.compute_distance(class1, class2)
                self.H[class1_idx, class2_idx] = distance
                self.H[class2_idx, class1_idx] = distance
        pickle.dump(self.H, open(os.path.join(self.data_dir, 'MCC/H_matrix_{}l_{}d_{}.p'.format(self.m, self.n,
                                                                                                self.add_none)), 'wb'))

    def generate_clusters_by_hierarchical_distance(self):
        """
        Generates clusters such that the labels are as hierarchically close as possible within the clusters,
        while maintaining mututal exclusivity within each cluster
        :return:
        """
        unordered_freqs = self.compute_label_freqs()  # dict of label: count
        freqs = sorted(unordered_freqs.items(), key=operator.itemgetter(1), reverse=True)  # list of tuples
        # [(code, freq), ...] in descending order of frequency
        max_count = max([d[1] for d in freqs]) if not self.max_freq_threshold else \
            round(self.n * self.max_freq_threshold)  # maximum absolute frequency we'll allow in order for a code
        # to make it in to a cluster

        self.loners = {d[0] for d in freqs if
                       d[1] > max_count}  # set of codes which are too high-freq for a cluster
        freqs = [f for f in freqs if f[0] not in self.loners]  # remove the loners from our frequency list
        remaining_labels = {s[0] for s in freqs}  # set of labels we can still cluster

        non_overlap_dict = {l: np.where(self.C[l, :] == 0)[0].tolist() for l in
                            remaining_labels}  # a dictionary for each
        # label, {label: [list of labels which the key label never co-occurs with], ...}

        print("Generating hierarchical clusters...")
        for entry in tqdm(freqs):  # iterate over every label we didn't discard due to high-freq
            seed = entry[
                0]  # since we're starting with the most high-freq labels, whichever label we're on is the seed
            labels = non_overlap_dict[seed]  # labels which do not overlap with our chosen seed
            if seed not in remaining_labels:  # if we already assigned this label to a cluster, it can't be a seed
                continue
            else:
                sorted_closeby_labels = np.argsort(self.H[seed, :])
                labels = [l for l in sorted_closeby_labels if l in labels and l in remaining_labels]  # gives us
                # elegible labels in ascending distance from our seed
                # print(self.mlb.classes_[seed])
                # print([self.mlb.classes_[l] for l in labels[:15]])
                cluster = []
                for lab in labels:  # iterate over eligible labels...
                    non_overlaps = set(
                        non_overlap_dict[lab])  # get the set of labels which current potential/eligible
                    # label doesn't co-occur with...
                    if set(cluster).issubset(non_overlaps):  # make sure that none of the labels currently in the
                        # cluster co-occur with this label
                        cluster.append(lab)
                    if len(
                            cluster) == self.max_cluster_size - 1:  # if we reached the maximum cluster size, we're done
                        break

                self.clusters[seed] = cluster + [seed]  # add to dict which keeps track of clusters by their seeds
                remaining_labels -= set(
                    self.clusters[seed])  # remove these clustered labels from our set of remaining
        for l in self.loners:  # make sure to add the loners which we discarded due to high freq; they simply form
            # their own cluster
            self.clusters[l] = [l]


        try:
            temp = self.clusters.deepcopy()
            to_remove = set()
            # now, we want to find those left over, low-frequency codes which ended up alone in their own cluster; the idea
            # is to find a cluster we could add them to, which may cause the max_cluster_size to be exceeded by 1 or 2,
            # but I feel it's better than leaving them on their own, if we want to employ an LDAM loss function
            # sometimes, this doesn't work though and then certain labels are left without a cluster, thus the
            # try/except
            for seed, cluster in self.clusters.items():
                if len(cluster) < self.min_cluster_size and seed not in self.loners:
                    for label in cluster:
                        non_overlaps = set(non_overlap_dict[label])
                        for seed2, cluster2 in self.clusters.items():
                            # if label == seed2 or seed2 in self.loners:
                            if seed == seed2 or seed2 in self.loners:
                                continue
                            else:
                                if set(cluster2).issubset(non_overlaps):
                                    temp[seed2].append(label)
                                    to_remove.add(seed)
                                    break
            for r in to_remove:
                del temp[r]
            assert len(self.mlb.classes_) == len([l for label_list in self.clusters.values() for l in label_list]), \
                "The following labels are not in clusters: {}".format(set(self.mlb.classes_) -
                                                                      set([self.mlb.classes_[l] for label_list in
                                                                           self.clusters.values() for l in label_list]))
            self.clusters = temp
        except:
            pass

        assert len(self.mlb.classes_) == len([l for label_list in self.clusters.values() for l in label_list]), \
            "The following labels are not in clusters: {}".format(set(self.mlb.classes_) -
                                                                  set([self.mlb.classes_[l] for label_list in
                                                                       self.clusters.values() for l in label_list]))

    def split_data_by_clusters(self, data_type):
        """
        Splits data such that each cluster (a separate MCC task) has its own dataset
        :return:
        """
        self.data_type = data_type
        self.load_data(data_type)
        if data_type == 'train':
            self.generate_clusters_by_hierarchical_distance()
            data_type = self.train_data
        elif data_type == 'dev':
            data_type = self.dev_data

        mlb = pickle.load(open(os.path.join(self.data_dir, 'mlb_0_False.p'), 'rb'))
        data_dict = defaultdict(list)
        self.make_cluster_idx2general_idx_dict()

        for item in data_type:
            doc_id = item[0]
            text = item[1]
            label_matrix = item[2]
            labels = np.nonzero(label_matrix)[0].tolist()
            activated_clusters = [seed for seed, cluster in self.clusters.items() if bool(set(labels) & set(cluster))]
            if self.data_type == 'train':
                assert len(labels) == len(
                    activated_clusters), "Sorry, mismatch with labels {} and activated clusters {}, " \
                                         "document {}".format('|'.join([str(l) for l in labels]),
                                                              '|'.join([str(a) for a in
                                                                        activated_clusters]),
                                                              doc_id)

            for seed in activated_clusters:
                label_matrix = np.zeros(len(self.clusters[seed]))  # generate an empty matrix of cluster size to add
                # binary labels to for MLCC using BR
                filtered_labels = set(labels) & set(self.clusters[seed])  # determine which labels of this example are
                # relevant to the currently activated cluster
                new_cluster_specific_label_idx = [self.overall_idx2cluster_idx[seed][l] for l in filtered_labels]
                label_matrix[new_cluster_specific_label_idx] = 1
                # if self.data_type == 'train':
                #     print(label_matrix)
                data_dict[seed].append((doc_id, text, label_matrix))

        for seed, data in data_dict.items():
            if not os.path.exists(os.path.join(self.out_dir, str(seed))):
                os.mkdir(os.path.join(self.out_dir, str(seed)))
            with open(os.path.join(self.out_dir, str(seed), self.data_type + '.tsv'), 'w') as f:
                file_name = os.path.join(str(seed), '{}_doc_id2gold.p'.format(self.data_type))
                temp = dict()
                if self.data_type == 'train':
                    all_labs = np.array([e[-1] for e in data])
                    self.label_class_counts[seed] = np.sum(all_labs, axis=0).tolist()
                for entry in data:
                    # if self.data_type == 'train':
                    #     lab_idx = np.nonzero(entry[-1])[0].item()
                    #     self.label_class_counts[seed][lab_idx] += 1  # for class counts

                    self.doc_id2activated_clusters[entry[0]].add(seed)
                    temp[entry[0]] = entry[2]
                    lab = self.cluster_idx2overall_idx[seed][np.nonzero(entry[2])[0][0]]
                    lab = mlb.classes_[lab]
                    f.write(str(entry[0]) + '\t' + entry[1] + '\t' + lab + '\n')

                pickle.dump(temp, open(os.path.join(self.out_dir, file_name), 'wb'))
                pickle.dump(self.overall_idx2cluster_idx[seed],
                            open(os.path.join(self.out_dir, str(seed), 'overall_idx2cluster_idx.p'), 'wb'))
                pickle.dump(self.cluster_idx2overall_idx[seed],
                            open(os.path.join(self.out_dir, str(seed), 'cluster_idx2overall_idx.p'), 'wb'))

    def split_into_cluster_prediction_data(self):
        """
        Creates data splits for the MLCC task
        :return:
        """
        self.seed2cluster_idx = {c: i for i, c in enumerate(self.clusters.keys())}
        self.cluster_idx2seed = {i: c for i, c in enumerate(self.clusters.keys())}
        temp = []

        with open(os.path.join(self.out_dir, 'doc_ids2clusters.p'), 'wb') as pickle_f, \
                open(os.path.join(self.out_dir, 'doc_ids2clusters.tsv'), 'w') as f:
            for doc_id in self.train_ids:
                activated_clusters = self.doc_id2activated_clusters[doc_id]
                temp.append(len(activated_clusters))
                vectorized_activated_clusters = np.zeros(len(self.clusters))
                vectorized_activated_clusters[[self.seed2cluster_idx[i] for i in activated_clusters]] = 1
                self.doc_id2activated_clusters[doc_id] = vectorized_activated_clusters
                f.write(doc_id + '\t' + '|'.join([str(a) for a in activated_clusters]) + '\n')
            for doc_id in self.dev_ids:
                activated_clusters = self.doc_id2activated_clusters[doc_id]
                temp.append(len(activated_clusters))
                vectorized_activated_clusters = np.zeros(len(self.clusters))
                vectorized_activated_clusters[[self.seed2cluster_idx[i] for i in activated_clusters]] = 1
                self.doc_id2activated_clusters[doc_id] = vectorized_activated_clusters
                f.write(str(doc_id) + '\t' + '|'.join([str(a) for a in activated_clusters]) + '\n')
            pickle.dump(self.doc_id2activated_clusters, pickle_f)

        pickle.dump(self.seed2cluster_idx, open(os.path.join(self.out_dir, 'seed2cluster_idx.p'), 'wb'))
        pickle.dump(self.cluster_idx2seed, open(os.path.join(self.out_dir, 'cluster_idx2seed.p'), 'wb'))
        print("Average number of activated clusters per input doc:", np.mean(temp))
        print("Total number of clusters:", len(self.clusters))

    def main(self):
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
            self.split_data_by_clusters('train')
            self.split_data_by_clusters('dev')
            self.split_into_cluster_prediction_data()
            self.generate_preliminary_exp_data()
            self.compute_class_counts()
            pickle.dump(self.clusters, open(os.path.join(self.out_dir, 'clusters.p'), 'wb'))
        else:
            self.clusters = pickle.load(open(os.path.join(self.out_dir, 'clusters.p'), 'rb'))


class MC2CFuzzyLabelClusterer():
    def __init__(self, data_dir, max_cluster_size=10, min_cluster_size=5, max_freq_threshold=0.0, add_none=False):
        self.data_dir = data_dir
        self.data_type = ''
        self.load_data('train')
        self.load_data('dev')
        self.add_none = add_none
        self.train_ids = [d[0] for d in self.train_data]
        self.dev_ids = [d[0] for d in self.dev_data]
        self.max_freq_threshold = max_freq_threshold
        self.m = len(self.train_data[0][2])  # num labels
        self.n = len(self.train_data)  # num docs
        self.E = np.zeros((self.n, self.m))
        self.C = np.zeros((self.m, self.m))
        self.create_E_matrix()
        self.power_dict = dict()  # per-label clustering power dictionary
        self.clusters = defaultdict(set)  # dict of {seed: {label1, ..., labeln},...}
        self.loners = set()  # those labels which get their own single binary classifier
        self.doc_id2activated_clusters = defaultdict(set)  # dictionary mapping doc_id --> relevant clusters
        self.max_cluster_size = max_cluster_size
        self.min_cluster_size = min_cluster_size
        self.n_c = (self.m // (self.max_cluster_size - 1)) + 1
        self.cluster_class_counts = dict()
        self.label_class_counts = dict()
        self.overall_idx2cluster_idx = defaultdict(dict)
        self.cluster_idx2overall_idx = defaultdict(dict)

        if not os.path.exists(os.path.join(self.data_dir, 'MCC/')):
            os.mkdir(os.path.join(self.data_dir, 'MCC/'))
        try:
            self.C = pickle.load(
                open(os.path.join(self.data_dir, 'MCC/C_matrix_{}l_{}d_{}.p'.format(self.m, self.n, self.add_none)),
                     'rb'))
        except:
            self.compute_C_matrix()

        self.delta = self.n_c / self.m
        self.epsilon = 0.001

    def load_data(self, data_type):
        self.train_data = pickle.load(open(os.path.join(self.data_dir, 'train_0_False.p'.format(data_type)), 'rb'))
        self.dev_data = pickle.load(open(os.path.join(self.data_dir, 'dev_0_False.p'.format(data_type)), 'rb'))

    def create_E_matrix(self):
        try:
            for i, (doc_id, text, labels, ranks) in enumerate(self.train_data):
                # labels = np.append(labels, np.array([0]))
                self.E[i, :] = labels
        except:
            for i, (doc_id, text, labels) in enumerate(self.train_data):
                # labels = np.append(labels, np.array([0]))
                self.E[i, :] = labels

    def compute_C_matrix(self):
        """
        Creates a co-occurrence matrix from self.E
        :return:
        """
        for i in trange(self.m):
            for j in range(self.n):
                co_occurrences = set(np.nonzero(self.E[j, :])[0])
                if i not in co_occurrences:
                    continue
                else:
                    for l in co_occurrences:
                        self.C[i, l] = 1
        pickle.dump(self.C, open(os.path.join(self.data_dir, 'MCC/C_matrix_{}l_{}d_{}.p'.format(self.m, self.n,
                                                                                                self.add_none)), 'wb'))

    def generate_clusters_by_freq(self):
        """
        Generates clusters such that the frequencies are as close as possible within the clusters, while maintaining
        mututal exclusivity within each cluster
        :return:
        """
        unordered_freqs = self.compute_label_freqs()  # dict of label: count
        freqs = sorted(unordered_freqs.items(), key=operator.itemgetter(1), reverse=True)  # list of tuples
        # [(code, freq), ...] in descending order of frequency
        max_count = max([d[1] for d in freqs]) if not self.max_freq_threshold else \
            round(self.n * self.max_freq_threshold)  # maximum absolute frequency we'll allow in order for a code
        # to make it in to a cluster

        self.loners = {d[0] for d in freqs if d[1] > max_count}  # set of codes which are too high-freq for a cluster
        freqs = [f for f in freqs if f[0] not in self.loners]  # remove the loners from our frequency list
        remaining_labels = {s[0] for s in freqs}  # set of labels we can still cluster

        non_overlap_dict = {l: np.where(self.C[l, :] == 0)[0].tolist() for l in
                            remaining_labels}  # a dictionary for each
        # label, {label: [list of labels which the key label never co-occurs with], ...}

        for entry in freqs:  # iterate over every label we didn't discard due to high-freq
            seed = entry[0]  # since we're starting with the most high-freq labels, whichever label we're on is the seed
            labels = non_overlap_dict[seed]  # labels which do not overlap with our chosen seed
            if seed not in remaining_labels:  # if we already assigned this label to a cluster, it can't be a seed
                continue
            else:
                labels = [l[0] for l in freqs if l[0] in labels and l[0] in remaining_labels]  # gives us the eligible
                #  labels in descending freq
                cluster = []
                for lab in labels:  # iterate over eligible labels...
                    non_overlaps = set(non_overlap_dict[lab])  # get the set of labels which current potential/eligible
                    # label doesn't co-occur with...
                    if set(cluster).issubset(non_overlaps):  # make sure that none of the labels currently in the
                        # cluster co-occur with this label
                        cluster.append(lab)
                    if len(cluster) == self.max_cluster_size - 1:  # if we reached the maximum cluster size, we're done
                        break

                self.clusters[seed] = cluster + [seed]  # add to dict which keeps track of clusters by their seeds
                remaining_labels -= set(self.clusters[seed])  # remove these clustered labels from our set of remaining
        for l in self.loners:  # make sure to add the loners which we discarded due to high freq; they simply form
            # their own cluster
            self.clusters[l] = [l]

        to_remove = set()
        # now, we want to find those left over, low-frequency codes which ended up alone in their own cluster; the idea
        # is to find a cluster we could add them to, which may cause the max_cluster_size to be exceeded by 1 or 2,
        # but I feel it's better than leaving them on their own, if we want to employ an LDAM loss function
        for seed, cluster in self.clusters.items():
            if len(cluster) < self.min_cluster_size and seed not in self.loners:
                for label in cluster:
                    non_overlaps = set(non_overlap_dict[label])
                    for seed2, cluster2 in self.clusters.items():
                        if label == seed2 or seed2 in self.loners:
                            continue
                        else:
                            if set(cluster2).issubset(non_overlaps):
                                self.clusters[seed2].append(label)
                                to_remove.add(seed)
                                break
        for r in to_remove:
            del self.clusters[r]

    def compute_label_freqs(self):
        label_freq_dict = {i: np.sum(self.E[:, i]) for i in range(self.m)}
        return label_freq_dict

    def make_cluster_idx2general_idx_dict(self):
        """
        Creates a dictionary which maps between the global index of a label and the local (inner-cluster-specific)
        index. The seed is what identifies a cluster, and this gives us a way to map between what labels are at
        what indices within a cluster
        :return:
        """

        for seed, cluster in self.clusters.items():
            for i, c in enumerate(cluster):
                self.overall_idx2cluster_idx[seed][c] = i  # {Seed: {mlb label idx: inner-cluster idx,...},...}
                self.cluster_idx2overall_idx[seed][i] = c  # {Seed: {inner-cluster idx: mlb label idx,...},...}

    def split_data_by_clusters(self, data_type):
        """
        Splits data such that each cluster (a separate MCC task) has its own dataset
        :return:
        """
        self.data_type = data_type
        self.load_data(data_type)
        if data_type == 'train':
            self.generate_clusters_by_freq()
            data_type = self.train_data
        elif data_type == 'dev':
            data_type = self.dev_data

        out_dir = os.path.join(self.data_dir, 'MCC/{}_{}_{}_{}'.format(self.min_cluster_size, self.max_cluster_size,
                                                                       self.max_freq_threshold, self.add_none))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        mlb = pickle.load(open(os.path.join(self.data_dir, 'mlb_0_False.p'), 'rb'))
        data_dict = defaultdict(list)
        self.make_cluster_idx2general_idx_dict()

        for item in data_type:
            doc_id = item[0]
            text = item[1]
            label_matrix = item[2]
            labels = np.nonzero(label_matrix)[0].tolist()
            activated_clusters = [seed for seed, cluster in self.clusters.items() if bool(set(labels) & set(cluster))]
            if self.data_type == 'train':
                assert len(labels) == len(
                    activated_clusters), "Sorry, mismatch with labels {} and activated clusters {}, " \
                                         "document {}".format('|'.join([str(l) for l in labels]),
                                                              '|'.join([str(a) for a in
                                                                        activated_clusters]),
                                                              doc_id)

            for seed in activated_clusters:
                label_matrix = np.zeros(len(self.clusters[seed]))  # generate an empty matrix of cluster size to add
                # binary labels to for MLCC using BR
                filtered_labels = set(labels) & set(self.clusters[seed])  # determine which labels of this example are
                # relevant to the currently activated cluster
                new_cluster_specific_label_idx = [self.overall_idx2cluster_idx[seed][l] for l in filtered_labels]
                label_matrix[new_cluster_specific_label_idx] = 1
                # if self.data_type == 'train':
                #     print(label_matrix)
                data_dict[seed].append((doc_id, text, label_matrix))

        for seed, data in data_dict.items():
            if not os.path.exists(os.path.join(out_dir, str(seed))):
                os.mkdir(os.path.join(out_dir, str(seed)))
            with open(os.path.join(out_dir, str(seed), self.data_type + '.tsv'), 'w') as f:
                file_name = os.path.join(str(seed), '{}_doc_id2gold.p'.format(self.data_type))
                temp = dict()
                if self.data_type == 'train':
                    all_labs = np.array([e[-1] for e in data])
                    self.label_class_counts[seed] = np.sum(all_labs, axis=0).tolist()
                for entry in data:
                    # if self.data_type == 'train':
                    #     lab_idx = np.nonzero(entry[-1])[0].item()
                    #     self.label_class_counts[seed][lab_idx] += 1  # for class counts

                    self.doc_id2activated_clusters[entry[0]].add(seed)
                    temp[entry[0]] = entry[2]
                    lab = self.cluster_idx2overall_idx[seed][np.nonzero(entry[2])[0][0]]
                    lab = mlb.classes_[lab]
                    f.write(str(entry[0]) + '\t' + entry[1] + '\t' + lab + '\n')

                pickle.dump(temp, open(os.path.join(out_dir, file_name), 'wb'))
                pickle.dump(self.overall_idx2cluster_idx[seed],
                            open(os.path.join(out_dir, str(seed), 'overall_idx2cluster_idx.p'), 'wb'))
                pickle.dump(self.cluster_idx2overall_idx[seed],
                            open(os.path.join(out_dir, str(seed), 'cluster_idx2overall_idx.p'), 'wb'))

    def split_into_cluster_prediction_data(self):
        self.seed2cluster_idx = {c: i for i, c in enumerate(self.clusters.keys())}
        self.cluster_idx2seed = {i: c for i, c in enumerate(self.clusters.keys())}
        temp = []

        out_dir = os.path.join(self.data_dir, 'MCC/{}_{}_{}_{}'.format(self.min_cluster_size, self.max_cluster_size,
                                                                       self.max_freq_threshold, self.add_none))
        with open(os.path.join(out_dir, 'doc_ids2clusters.p'), 'wb') as pickle_f, \
                open(os.path.join(out_dir, 'doc_ids2clusters.tsv'), 'w') as f:
            for doc_id in self.train_ids:
                activated_clusters = self.doc_id2activated_clusters[doc_id]
                temp.append(len(activated_clusters))
                vectorized_activated_clusters = np.zeros(len(self.clusters))
                vectorized_activated_clusters[[self.seed2cluster_idx[i] for i in activated_clusters]] = 1
                self.doc_id2activated_clusters[doc_id] = vectorized_activated_clusters
                f.write(doc_id + '\t' + '|'.join([str(a) for a in activated_clusters]) + '\n')
            for doc_id in self.dev_ids:
                activated_clusters = self.doc_id2activated_clusters[doc_id]
                temp.append(len(activated_clusters))
                vectorized_activated_clusters = np.zeros(len(self.clusters))
                vectorized_activated_clusters[[self.seed2cluster_idx[i] for i in activated_clusters]] = 1
                self.doc_id2activated_clusters[doc_id] = vectorized_activated_clusters
                f.write(str(doc_id) + '\t' + '|'.join([str(a) for a in activated_clusters]) + '\n')
            pickle.dump(self.doc_id2activated_clusters, pickle_f)

        pickle.dump(self.seed2cluster_idx, open(os.path.join(out_dir, 'seed2cluster_idx.p'), 'wb'))
        pickle.dump(self.cluster_idx2seed, open(os.path.join(out_dir, 'cluster_idx2seed.p'), 'wb'))
        print("Average number of activated clusters per input doc:", np.mean(temp))
        print("Total number of clusters:", len(self.clusters))

    def generate_preliminary_exp_data(self):
        PRELIMINARY_EXP_DATA = []
        out_dir = os.path.join(self.data_dir, 'MCC/{}_{}_{}_{}'.format(self.min_cluster_size, self.max_cluster_size,
                                                                       self.max_freq_threshold, self.add_none))
        with open(os.path.join(out_dir, 'train.p'), 'wb') as pf:
            for d in self.train_data:
                PRELIMINARY_EXP_DATA.append((d[0], d[1], self.doc_id2activated_clusters[d[0]]))
            pickle.dump(PRELIMINARY_EXP_DATA, pf)

    def compute_class_counts(self):
        out_dir = os.path.join(self.data_dir, 'MCC/{}_{}_{}_{}'.format(self.min_cluster_size, self.max_cluster_size,
                                                                       self.max_freq_threshold, self.add_none))
        # label_class_counts = dict()
        # for seed, idx2count_dict in self.label_class_counts.items():
        #     label_class_counts[seed] = [idx2count_dict[k] for k in sorted(idx2count_dict)]
        # print(label_class_counts)

        cluster_class_counts = {seed: sum(counts) for seed, counts in self.label_class_counts.items()}

        cluster_class_counts = [cluster_class_counts[self.cluster_idx2seed[i]] for i in
                                range(len(cluster_class_counts))]

        cluster_class_counts = [(self.n - c) / c for c in cluster_class_counts]
        label_class_counts = {self.seed2cluster_idx[k]: v for k, v in self.label_class_counts.items()}
        # the inner-cluster class counts are now in a dictionary where they are looked up by their index in the
        # MLCC step, rather then by their seed
        pickle.dump(label_class_counts, open(os.path.join(out_dir, 'local_class_counts.p'), 'wb'))
        pickle.dump(cluster_class_counts, open(os.path.join(out_dir, 'global_cluster_counts.p'), 'wb'))

    def generate_activated_clusters_for_dev_data(self):
        print(self.clusters)
        pass

    def main(self):
        self.split_data_by_clusters('train')
        self.split_data_by_clusters('dev')
        self.split_into_cluster_prediction_data()
        self.generate_preliminary_exp_data()
        self.compute_class_counts()


class MC2CFuzzyHierarchicalLabelClusterer():
    def __init__(self, data_dir, max_cluster_size=10, min_cluster_size=5, max_freq_threshold=0.0, add_none=False):
        self.data_dir = data_dir
        self.data_type = ''
        self.load_data('train')
        self.load_data('dev')
        self.add_none = add_none
        self.train_ids = [d[0] for d in self.train_data]
        self.dev_ids = [d[0] for d in self.dev_data]
        self.max_freq_threshold = max_freq_threshold
        self.m = len(self.train_data[0][2])  # num labels
        self.n = len(self.train_data)  # num docs
        self.E = np.zeros((self.n, self.m))
        self.C = np.zeros((self.m, self.m))
        self.create_E_matrix()
        self.power_dict = dict()  # per-label clustering power dictionary
        self.clusters = defaultdict(set)  # dict of {seed: {label1, ..., labeln},...}
        self.loners = set()  # those labels which get their own single binary classifier
        self.doc_id2activated_clusters = defaultdict(set)  # dictionary mapping doc_id --> relevant clusters
        self.max_cluster_size = max_cluster_size
        self.min_cluster_size = min_cluster_size
        self.n_c = (self.m // (self.max_cluster_size - 1)) + 1
        self.cluster_class_counts = dict()
        self.label_class_counts = dict()
        self.overall_idx2cluster_idx = defaultdict(dict)
        self.cluster_idx2overall_idx = defaultdict(dict)

        if not os.path.exists(os.path.join(self.data_dir, 'MCC/')):
            os.mkdir(os.path.join(self.data_dir, 'MCC/'))
        try:
            self.C = pickle.load(
                open(os.path.join(self.data_dir, 'MCC/C_matrix_{}l_{}d_{}.p'.format(self.m, self.n, self.add_none)),
                     'rb'))
        except:
            self.compute_C_matrix()

        self.delta = self.n_c / self.m
        self.epsilon = 0.001

    def load_data(self, data_type):
        self.train_data = pickle.load(open(os.path.join(self.data_dir, 'train_0_False.p'.format(data_type)), 'rb'))
        self.dev_data = pickle.load(open(os.path.join(self.data_dir, 'dev_0_False.p'.format(data_type)), 'rb'))

    def create_E_matrix(self):
        try:
            for i, (doc_id, text, labels, ranks) in enumerate(self.train_data):
                # labels = np.append(labels, np.array([0]))
                self.E[i, :] = labels
        except:
            for i, (doc_id, text, labels) in enumerate(self.train_data):
                # labels = np.append(labels, np.array([0]))
                self.E[i, :] = labels

    def compute_C_matrix(self):
        """
        Creates a co-occurrence matrix from self.E
        :return:
        """
        for i in trange(self.m):
            for j in range(self.n):
                co_occurrences = set(np.nonzero(self.E[j, :])[0])
                if i not in co_occurrences:
                    continue
                else:
                    for l in co_occurrences:
                        self.C[i, l] = 1
        pickle.dump(self.C, open(os.path.join(self.data_dir, 'MCC/C_matrix_{}l_{}d_{}.p'.format(self.m, self.n,
                                                                                                self.add_none)), 'wb'))

    def generate_clusters_by_freq(self):
        """
        Generates clusters such that the frequencies are as close as possible within the clusters, while maintaining
        mututal exclusivity within each cluster
        :return:
        """
        unordered_freqs = self.compute_label_freqs()  # dict of label: count
        freqs = sorted(unordered_freqs.items(), key=operator.itemgetter(1), reverse=True)  # list of tuples
        # [(code, freq), ...] in descending order of frequency
        max_count = max([d[1] for d in freqs]) if not self.max_freq_threshold else \
            round(self.n * self.max_freq_threshold)  # maximum absolute frequency we'll allow in order for a code
        # to make it in to a cluster

        self.loners = {d[0] for d in freqs if d[1] > max_count}  # set of codes which are too high-freq for a cluster
        freqs = [f for f in freqs if f[0] not in self.loners]  # remove the loners from our frequency list
        remaining_labels = {s[0] for s in freqs}  # set of labels we can still cluster

        non_overlap_dict = {l: np.where(self.C[l, :] == 0)[0].tolist() for l in
                            remaining_labels}  # a dictionary for each
        # label, {label: [list of labels which the key label never co-occurs with], ...}

        for entry in freqs:  # iterate over every label we didn't discard due to high-freq
            seed = entry[0]  # since we're starting with the most high-freq labels, whichever label we're on is the seed
            labels = non_overlap_dict[seed]  # labels which do not overlap with our chosen seed
            if seed not in remaining_labels:  # if we already assigned this label to a cluster, it can't be a seed
                continue
            else:
                labels = [l[0] for l in freqs if l[0] in labels and l[0] in remaining_labels]  # gives us the eligible
                #  labels in descending freq
                cluster = []
                for lab in labels:  # iterate over eligible labels...
                    non_overlaps = set(non_overlap_dict[lab])  # get the set of labels which current potential/eligible
                    # label doesn't co-occur with...
                    if set(cluster).issubset(non_overlaps):  # make sure that none of the labels currently in the
                        # cluster co-occur with this label
                        cluster.append(lab)
                    if len(cluster) == self.max_cluster_size - 1:  # if we reached the maximum cluster size, we're done
                        break

                self.clusters[seed] = cluster + [seed]  # add to dict which keeps track of clusters by their seeds
                remaining_labels -= set(self.clusters[seed])  # remove these clustered labels from our set of remaining
        for l in self.loners:  # make sure to add the loners which we discarded due to high freq; they simply form
            # their own cluster
            self.clusters[l] = [l]

        to_remove = set()
        # now, we want to find those left over, low-frequency codes which ended up alone in their own cluster; the idea
        # is to find a cluster we could add them to, which may cause the max_cluster_size to be exceeded by 1 or 2,
        # but I feel it's better than leaving them on their own, if we want to employ an LDAM loss function
        for seed, cluster in self.clusters.items():
            if len(cluster) < self.min_cluster_size and seed not in self.loners:
                for label in cluster:
                    non_overlaps = set(non_overlap_dict[label])
                    for seed2, cluster2 in self.clusters.items():
                        if label == seed2 or seed2 in self.loners:
                            continue
                        else:
                            if set(cluster2).issubset(non_overlaps):
                                self.clusters[seed2].append(label)
                                to_remove.add(seed)
                                break
        for r in to_remove:
            del self.clusters[r]

    def compute_label_freqs(self):
        label_freq_dict = {i: np.sum(self.E[:, i]) for i in range(self.m)}
        return label_freq_dict

    def make_cluster_idx2general_idx_dict(self):
        """
        Creates a dictionary which maps between the global index of a label and the local (inner-cluster-specific)
        index. The seed is what identifies a cluster, and this gives us a way to map between what labels are at
        what indices within a cluster
        :return:
        """

        for seed, cluster in self.clusters.items():
            for i, c in enumerate(cluster):
                self.overall_idx2cluster_idx[seed][c] = i  # {Seed: {mlb label idx: inner-cluster idx,...},...}
                self.cluster_idx2overall_idx[seed][i] = c  # {Seed: {inner-cluster idx: mlb label idx,...},...}

    def split_data_by_clusters(self, data_type):
        """
        Splits data such that each cluster (a separate MCC task) has its own dataset
        :return:
        """
        self.data_type = data_type
        self.load_data(data_type)
        if data_type == 'train':
            self.generate_clusters_by_freq()
            data_type = self.train_data
        elif data_type == 'dev':
            data_type = self.dev_data

        out_dir = os.path.join(self.data_dir, 'MCC/{}_{}_{}_{}'.format(self.min_cluster_size, self.max_cluster_size,
                                                                       self.max_freq_threshold, self.add_none))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        mlb = pickle.load(open(os.path.join(self.data_dir, 'mlb_0_False.p'), 'rb'))
        data_dict = defaultdict(list)
        self.make_cluster_idx2general_idx_dict()

        for item in data_type:
            doc_id = item[0]
            text = item[1]
            label_matrix = item[2]
            labels = np.nonzero(label_matrix)[0].tolist()
            activated_clusters = [seed for seed, cluster in self.clusters.items() if bool(set(labels) & set(cluster))]
            if self.data_type == 'train':
                assert len(labels) == len(
                    activated_clusters), "Sorry, mismatch with labels {} and activated clusters {}, " \
                                         "document {}".format('|'.join([str(l) for l in labels]),
                                                              '|'.join([str(a) for a in
                                                                        activated_clusters]),
                                                              doc_id)

            for seed in activated_clusters:
                label_matrix = np.zeros(len(self.clusters[seed]))  # generate an empty matrix of cluster size to add
                # binary labels to for MLCC using BR
                filtered_labels = set(labels) & set(self.clusters[seed])  # determine which labels of this example are
                # relevant to the currently activated cluster
                new_cluster_specific_label_idx = [self.overall_idx2cluster_idx[seed][l] for l in filtered_labels]
                label_matrix[new_cluster_specific_label_idx] = 1
                # if self.data_type == 'train':
                #     print(label_matrix)
                data_dict[seed].append((doc_id, text, label_matrix))

        for seed, data in data_dict.items():
            if not os.path.exists(os.path.join(out_dir, str(seed))):
                os.mkdir(os.path.join(out_dir, str(seed)))
            with open(os.path.join(out_dir, str(seed), self.data_type + '.tsv'), 'w') as f:
                file_name = os.path.join(str(seed), '{}_doc_id2gold.p'.format(self.data_type))
                temp = dict()
                if self.data_type == 'train':
                    all_labs = np.array([e[-1] for e in data])
                    self.label_class_counts[seed] = np.sum(all_labs, axis=0).tolist()
                for entry in data:
                    # if self.data_type == 'train':
                    #     lab_idx = np.nonzero(entry[-1])[0].item()
                    #     self.label_class_counts[seed][lab_idx] += 1  # for class counts

                    self.doc_id2activated_clusters[entry[0]].add(seed)
                    temp[entry[0]] = entry[2]
                    lab = self.cluster_idx2overall_idx[seed][np.nonzero(entry[2])[0][0]]
                    lab = mlb.classes_[lab]
                    f.write(str(entry[0]) + '\t' + entry[1] + '\t' + lab + '\n')

                pickle.dump(temp, open(os.path.join(out_dir, file_name), 'wb'))
                pickle.dump(self.overall_idx2cluster_idx[seed],
                            open(os.path.join(out_dir, str(seed), 'overall_idx2cluster_idx.p'), 'wb'))
                pickle.dump(self.cluster_idx2overall_idx[seed],
                            open(os.path.join(out_dir, str(seed), 'cluster_idx2overall_idx.p'), 'wb'))

    def split_into_cluster_prediction_data(self):
        self.seed2cluster_idx = {c: i for i, c in enumerate(self.clusters.keys())}
        self.cluster_idx2seed = {i: c for i, c in enumerate(self.clusters.keys())}
        temp = []

        out_dir = os.path.join(self.data_dir, 'MCC/{}_{}_{}_{}'.format(self.min_cluster_size, self.max_cluster_size,
                                                                       self.max_freq_threshold, self.add_none))
        with open(os.path.join(out_dir, 'doc_ids2clusters.p'), 'wb') as pickle_f, \
                open(os.path.join(out_dir, 'doc_ids2clusters.tsv'), 'w') as f:
            for doc_id in self.train_ids:
                activated_clusters = self.doc_id2activated_clusters[doc_id]
                temp.append(len(activated_clusters))
                vectorized_activated_clusters = np.zeros(len(self.clusters))
                vectorized_activated_clusters[[self.seed2cluster_idx[i] for i in activated_clusters]] = 1
                self.doc_id2activated_clusters[doc_id] = vectorized_activated_clusters
                f.write(doc_id + '\t' + '|'.join([str(a) for a in activated_clusters]) + '\n')
            for doc_id in self.dev_ids:
                activated_clusters = self.doc_id2activated_clusters[doc_id]
                temp.append(len(activated_clusters))
                vectorized_activated_clusters = np.zeros(len(self.clusters))
                vectorized_activated_clusters[[self.seed2cluster_idx[i] for i in activated_clusters]] = 1
                self.doc_id2activated_clusters[doc_id] = vectorized_activated_clusters
                f.write(str(doc_id) + '\t' + '|'.join([str(a) for a in activated_clusters]) + '\n')
            pickle.dump(self.doc_id2activated_clusters, pickle_f)

        pickle.dump(self.seed2cluster_idx, open(os.path.join(out_dir, 'seed2cluster_idx.p'), 'wb'))
        pickle.dump(self.cluster_idx2seed, open(os.path.join(out_dir, 'cluster_idx2seed.p'), 'wb'))
        print("Average number of activated clusters per input doc:", np.mean(temp))
        print("Total number of clusters:", len(self.clusters))

    def generate_preliminary_exp_data(self):
        PRELIMINARY_EXP_DATA = []
        out_dir = os.path.join(self.data_dir, 'MCC/{}_{}_{}_{}'.format(self.min_cluster_size, self.max_cluster_size,
                                                                       self.max_freq_threshold, self.add_none))
        with open(os.path.join(out_dir, 'train.p'), 'wb') as pf:
            for d in self.train_data:
                PRELIMINARY_EXP_DATA.append((d[0], d[1], self.doc_id2activated_clusters[d[0]]))
            pickle.dump(PRELIMINARY_EXP_DATA, pf)

    def compute_class_counts(self):
        out_dir = os.path.join(self.data_dir, 'MCC/{}_{}_{}_{}'.format(self.min_cluster_size, self.max_cluster_size,
                                                                       self.max_freq_threshold, self.add_none))
        # label_class_counts = dict()
        # for seed, idx2count_dict in self.label_class_counts.items():
        #     label_class_counts[seed] = [idx2count_dict[k] for k in sorted(idx2count_dict)]
        # print(label_class_counts)

        cluster_class_counts = {seed: sum(counts) for seed, counts in self.label_class_counts.items()}

        cluster_class_counts = [cluster_class_counts[self.cluster_idx2seed[i]] for i in
                                range(len(cluster_class_counts))]

        cluster_class_counts = [(self.n - c) / c for c in cluster_class_counts]
        label_class_counts = {self.seed2cluster_idx[k]: v for k, v in self.label_class_counts.items()}
        # the inner-cluster class counts are now in a dictionary where they are looked up by their index in the
        # MLCC step, rather then by their seed
        pickle.dump(label_class_counts, open(os.path.join(out_dir, 'local_class_counts.p'), 'wb'))
        pickle.dump(cluster_class_counts, open(os.path.join(out_dir, 'global_cluster_counts.p'), 'wb'))

    def generate_activated_clusters_for_dev_data(self):
        print(self.clusters)
        pass

    def main(self):
        self.split_data_by_clusters('train')
        self.split_data_by_clusters('dev')
        self.split_into_cluster_prediction_data()
        self.generate_preliminary_exp_data()
        self.compute_class_counts()


if __name__ == '__main__':
    data_dir = 'processed_data/german/'
    # data_dir = 'processed_data/cantemist/'
    # data_dir = 'processed_data/spanish/es/'

    # clusterer = MC2CLabelClusterer(data_dir, max_cluster_size=10, min_cluster_size=5, max_freq_threshold=0.25)
    clusterer = MC2CHierarchicalLabelClusterer(data_dir, max_cluster_size=15, min_cluster_size=5,
                                               max_freq_threshold=0.25)

    clusterer.main()
    label_freqs = clusterer.compute_label_freqs()
    for k, v in clusterer.clusters.items():
        print(label_freqs[k], '---', [label_freqs[l] for l in v])
        print(clusterer.mlb.classes_[k], '---', [clusterer.mlb.classes_[l] for l in v])

    # print(clusterer.cluster_idx2seed[2])
    # print(clusterer.cluster_idx2seed[3])
    # print(clusterer.cluster_idx2seed[4])

    #
    # clusterer.generate_clusters_by_freq()
    # all_labels = []
    # for k, v in clusterer.clusters.items():
    #     print(k, '---', v)

    #     # all_labels.append(k)
    #     all_labels += v
    #
    # print(len(all_labels))
    # print(len(set(all_labels)))
    # # print(len(clusterer.clusters[11]))
    # # print(mlb.classes_[1])
    # #
    # label_freqs = clusterer.compute_label_freqs()
    # for k, v in clusterer.clusters.items():
    #     print(label_freqs[k], '---', [label_freqs[l] for l in v])
    #
    # print(clusterer.C[1,55])
    # print(all_labels)

    # seed = 7
    # for seed in clusterer.determine_seeds():
    #     new_E = clusterer.E[list(clusterer.clusters[seed]), :]
    #
    #     new_array = [tuple(row) for row in new_E.T]
    #     counts = Counter(new_array)
    #     print("total unique combos of seed {}:".format(str(seed)), len(set(new_array)))
    #     print(counts.values())
    #     print('-' * 100)

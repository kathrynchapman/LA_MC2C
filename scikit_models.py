import os, sys
import argparse
# from torch import load
import glob
import pickle
import copy
import time
from datetime import datetime
from random import shuffle
from tqdm import tqdm, trange

from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain
from skmultilearn.ensemble import RakelD, RakelO
from skmultilearn.ensemble.partition import LabelSpacePartitioningClassifier
from skmultilearn.ensemble.voting import MajorityVotingClassifier
from skmultilearn.cluster.random import RandomLabelSpaceClusterer
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
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from skmultilearn.dataset import load_dataset

"""
The below models are the original models from scikit-MultiLearn, with simple adjustments to support tqdm
status bar. Useful especially for the Rakel* models, which take a substantial time to train with large
label spaces/large data sets

"""


class MyBinaryRelevance(BinaryRelevance):
    def __init__(self, classifier=None, require_dense=None):
        super(MyBinaryRelevance, self).__init__(classifier, require_dense)

    def fit(self, X, y):
        """Fits classifier to training data
        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
        Returns
        -------
        self
            fitted instance of self
        Notes
        -----
        .. note :: Input matrices are converted to sparse format internally if a numpy representation is passed
        """
        X = self._ensure_input_format(
            X, sparse_format='csr', enforce_sparse=True)
        y = self._ensure_output_format(
            y, sparse_format='csc', enforce_sparse=True)

        self.classifiers_ = []
        self._generate_partition(X, y)
        self._label_count = y.shape[1]

        for i in trange(self.model_count_):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self._generate_data_subset(y, self.partition_[i], axis=1)
            if issparse(y_subset) and y_subset.ndim > 1 and y_subset.shape[1] == 1:
                y_subset = np.ravel(y_subset.toarray())
            classifier.fit(self._ensure_input_format(
                X), self._ensure_output_format(y_subset))
            self.classifiers_.append(classifier)

        return self

    def predict(self, X):
        """Predict labels for X
        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        Returns
        -------
        :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
        """
        predictions = [self._ensure_multi_label_from_single_class(
            self.classifiers_[label].predict(self._ensure_input_format(X)))
            for label in trange(self.model_count_)]

        return hstack(predictions)


class MyClassifierChain(ClassifierChain):
    def __init__(self, classifier=None, require_dense=None, order=None):
        super(MyClassifierChain, self).__init__(classifier, require_dense)

    def fit(self, X, y, order=None):
        """Fits classifier to training data
        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
        Returns
        -------
        self
            fitted instance of self
        Notes
        -----
        .. note :: Input matrices are converted to sparse format internally if a numpy representation is passed
        """

        # fit L = len(y[0]) BR classifiers h_i
        # on X + y[:i] as input space and y[i+1] as output

        X_extended = self._ensure_input_format(X, sparse_format='csc', enforce_sparse=True)
        y = self._ensure_output_format(y, sparse_format='csc', enforce_sparse=True)

        self._label_count = y.shape[1]
        self.classifiers_ = [None for x in range(self._label_count)]

        for label in tqdm(self._order()):
            self.classifier = copy.deepcopy(self.classifier)
            y_subset = self._generate_data_subset(y, label, axis=1)

            self.classifiers_[label] = self.classifier.fit(self._ensure_input_format(
                X_extended), self._ensure_output_format(y_subset))
            X_extended = hstack([X_extended, y_subset])

        return self

    def predict(self, X):
        """Predict labels for X
        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        Returns
        -------
        :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
        """

        X_extended = self._ensure_input_format(
            X, sparse_format='csc', enforce_sparse=True)

        for label in tqdm(self._order()):
            prediction = self.classifiers_[label].predict(
                self._ensure_input_format(X_extended))
            prediction = self._ensure_multi_label_from_single_class(prediction)
            X_extended = hstack([X_extended, prediction])
        return X_extended[:, -self._label_count:]


class MyLabelSpacePartitioningClassifier(MyBinaryRelevance):
    def __init__(self, classifier=None, clusterer=None, require_dense=None):
        super(MyLabelSpacePartitioningClassifier, self).__init__(classifier, require_dense)
        self.clusterer = clusterer
        self.copyable_attrs = ['clusterer', 'classifier', 'require_dense']

    def predict(self, X):
        """Predict labels for X
        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`
        Returns
        -------
        scipy.sparse of int
            binary indicator matrix with label assignments with shape
            :code:`(n_samples, n_labels)`
        """
        X = self._ensure_input_format(
            X, sparse_format='csr', enforce_sparse=True)
        result = sparse.lil_matrix((X.shape[0], self._label_count), dtype=int)

        for model in trange(self.model_count_):
            predictions = self._ensure_output_format(self.classifiers_[model].predict(
                X), sparse_format=None, enforce_sparse=True).nonzero()
            for row, column in zip(predictions[0], predictions[1]):
                result[row, self.partition_[model][column]] = 1

        return result

    def _generate_partition(self, X, y):
        """Cluster the label space
        Saves the partiton generated by the clusterer to :code:`self.partition_` and
        sets :code:`self.model_count_` to number of clusers and :code:`self._label_count`
        to number of labels.
        Parameters
        -----------
        X : numpy.ndarray or scipy.sparse
            input features of shape :code:`(n_samples, n_features)`, passed to clusterer
        y : numpy.ndarray or scipy.sparse
            binary indicator matrix with label assigments of shape
            :code:`(n_samples, n_labels)`
        Returns
        -------
        LabelSpacePartitioningClassifier
            returns an instance of itself
        """

        self.partition_ = self.clusterer.fit_predict(X, y)
        self.model_count_ = len(self.partition_)
        self._label_count = y.shape[1]

        return self


class MyMajorityVotingClassifier(MyLabelSpacePartitioningClassifier):
    def __init__(self, classifier=None, clusterer=None, require_dense=None):
        super(MyMajorityVotingClassifier, self).__init__(
            classifier=classifier, clusterer=clusterer, require_dense=require_dense
        )

    def predict(self, X):
        """Predict label assignments for X
        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`
        Returns
        -------
        scipy.sparse of float
            binary indicator matrix with label assignments with shape
            :code:`(n_samples, n_labels)`
        """
        predictions = [
            self._ensure_input_format(self._ensure_input_format(
                c.predict(X)), sparse_format='csc', enforce_sparse=True)
            for c in tqdm(self.classifiers_)
        ]

        voters = np.zeros(self._label_count, dtype='int')
        votes = sparse.lil_matrix(
            (predictions[0].shape[0], self._label_count), dtype='int')
        for model in trange(self.model_count_):
            for label in range(len(self.partition_[model])):
                votes[:, self.partition_[model][label]] = votes[
                                                          :, self.partition_[model][label]] + predictions[model][:,
                                                                                              label]
                voters[self.partition_[model][label]] += 1

        nonzeros = votes.nonzero()
        for row, column in zip(nonzeros[0], nonzeros[1]):
            votes[row, column] = np.round(
                votes[row, column] / float(voters[column]))

        return self._ensure_output_format(votes, enforce_sparse=False)

    def predict_proba(self, X):
        raise NotImplemented("The voting scheme does not define a method for calculating probabilities")


class MyRakelO(RakelO):
    def __init__(self, base_classifier=None, model_count=None, labelset_size=3, base_classifier_require_dense=None):
        super(MyRakelO, self).__init__()
        self.model_count = model_count
        self.labelset_size = labelset_size
        self.base_classifier = base_classifier
        self.base_classifier_require_dense = base_classifier_require_dense
        self.copyable_attrs = ['model_count', 'labelset_size',
                               'base_classifier_require_dense',
                               'base_classifier']

    def fit(self, X, y):
        """Fits classifier to training data
        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
        Returns
        -------
        self
            fitted instance of self
        """
        self.classifier = MyMajorityVotingClassifier(
            classifier=LabelPowerset(
                classifier=self.base_classifier,
                require_dense=self.base_classifier_require_dense
            ),
            clusterer=RandomLabelSpaceClusterer(
                cluster_size=self.labelset_size,
                cluster_count=self.model_count,
                allow_overlap=True
            ),
            require_dense=[False, False]
        )
        return self.classifier.fit(X, y)


class MyRakelD(RakelD):
    def __init__(self, base_classifier=None, labelset_size=3, base_classifier_require_dense=None):
        super(MyRakelD, self).__init__()
        self.labelset_size = labelset_size
        self.base_classifier = base_classifier
        self.base_classifier_require_dense = base_classifier_require_dense
        self.copyable_attrs = ['base_classifier', 'base_classifier_require_dense', 'labelset_size']

    def fit(self, X, y):
        """Fit classifier to multi-label data
        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse
            input features, can be a dense or sparse matrix of size
            :code:`(n_samples, n_features)`
        y : numpy.ndaarray or scipy.sparse {0,1}
            binary indicator matrix with label assignments, shape
            :code:`(n_samples, n_labels)`
        Returns
        -------
        fitted instance of self
        """
        self._label_count = y.shape[1]
        self.model_count_ = int(np.ceil(self._label_count / self.labelset_size))
        self.classifier_ = MyLabelSpacePartitioningClassifier(
            classifier=LabelPowerset(
                classifier=self.base_classifier,
                require_dense=self.base_classifier_require_dense
            ),
            clusterer=RandomLabelSpaceClusterer(
                cluster_size=self.labelset_size,
                cluster_count=self.model_count_,
                allow_overlap=False
            ),
            require_dense=[False, False]
        )
        return self.classifier_.fit(X, y)
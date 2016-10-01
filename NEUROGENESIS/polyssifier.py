"""This module computes the baseline results by applying various classifiers.
The classifiers used here are nearest neighbors, linear SVM, RBF SVM, decision
tree, random forest, logistic regression, naive bayes, and LDA.
equivalent to latest version by PB polyssifier_pb_12_23_15c
"""

__author__ = "Sergey Plis"
__copyright__ = "Copyright 2015, Mind Research Network"
__credits__ = ["Sergey Plis, Devon Hjelm, Alvaro Ulloa"]
__licence__ = "3-clause BSD"
__email__ = "splis@gmail.com"
__maintainer__ = "Sergey Plis"

#Edited by POlo 4/29/2016
# Removed LDA parameters (irrelevant for binary classification)


# Edited by Polo 4/27/2016
# Added dummy classifier for random performance calculation.
# Added list of distinct colors for plotting many classifiers
# Added preallocated parallleized variable ranking
# Added or fixed paralellized classification across folds
# Allowed RandomizedSeachCV to be used, done in parallel or serially


# Edited by Pouya, 12/23/2015
# Changed the plot variables to FN and FP errors.

# Edited by Pouya, 10/19/2015
# Added L1-regularization to logistic regression classifier

# Edited by Pouya, 09/21/2015
# Added variable ranking by t-test p-values

# Edited by Pouya, 09/11/2015
# Fixed the gray background color (Seaborn module conflict)
# Improved details in plotting

# Edited by Pouya, 08/24/2015
# Changed the plotting functions.

# Edited by Pouya, 08/11/2015:
# Added function to save the error results for each classifier inside the loop
# Fixed the bug with legend cut out inf saved figure
# Added filename as a title in figure

# Edited by Pouya, 08/10/2015:
# Added a control for detecting and discarding NaN values.
# Added a fix to capture not-converging error in LDA
# Saving selected numTopVars and classifier NAMES in the .mat file

# Edited by Pouya, 08/04/2015:
# Added variable selection

# Edited by Pouya, 07/29/2015:
# Changed the classifier output to confusion matrix
# Changed the save output file to .mat

# Edited by Pouya, 07/22/2015:
# Changed input format from .npy to .mat
# Changed Xvalidation from Stratified to Regular


USEJOBLIB=False

import argparse
import functools
from glob import glob
import logging

if USEJOBLIB:
    from joblib.pool import MemmapingPool as Pool
    from joblib.pool import ArrayMemmapReducer as Array
else:
    from multiprocessing import Pool
    from multiprocessing import Array

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as pl
import multiprocessing
import numpy as np
import os
from os import path
# import pandas as pd
import pickle
import random as rndc
from scipy.io import savemat
import scipy.io
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, ttest_ind

from sklearn.metrics import auc, mutual_info_score
from sklearn.metrics import roc_curve as rc
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
# from sklearn.qda import QDA
from sklearn.dummy import DummyClassifier
# from lightning.classification import CDClassifier, SDCAClassifier
import sys
import copy
# import seaborn as sns
import time
import json


# Font sizes for plotting
font = {'family' : 'normal',
        'size'   : '22'}
mpl.rc('font', **font)
FONT_SIZE = 22
mpl.rcParams['pdf.fonttype']=42
# please set this number to no more than the number of cores on the machine you're
# going to be running it on but high enough to help the computation
PROCESSORS = 1 # 12 # 31 #12 # 31 #
seed = rndc.SystemRandom().seed()
NAMES = ["Chance", "Nearest Neighbors", "Linear SVM", "RBF SVM",  "Decision Tree",
         "Random Forest", "Logistic Regression", "Naive Bayes", "LDA"]
#NAMES = ["Nearest Neighbors", "Linear SVM", "Decision Tree",
#         "Random Forest", "Logistic Regression", "Naive Bayes", "LDA"]
# NAMES = ['Logistic Regression']

def rank_vars(xTrain, yTrain, scoreFunc):
    """
    ranks variables using the supplied scoreFunc.
    Inputs:
        xTrain: training features matrix
        yTrain: training lables vector
        scoreFunc: function used to rank features (pearsonr or mutual_info_score)
    Output:
        returns the ranked list of features indices
    """
    funcsDic = {
        'pearsonr': [np.arange(xTrain.shape[1]), 1], 
        'mutual_info_score': np.arange(xTrain.shape[0]),
        'ttest_ind': [np.arange(xTrain.shape[1]), 1], 
        }
    
    scores = list()
    for feat in np.arange(xTrain.shape[1]):
        if scoreFunc.func_name == 'pearsonr':
            scores.append(scoreFunc(xTrain[:, feat], yTrain))
        elif scoreFunc.func_name == 'ttest_ind':
            scores.append(scoreFunc(xTrain[yTrain == 1, feat], xTrain[yTrain==-1, feat]))
    
    scores = np.asarray(scores)
    pvals = scores[funcsDic[scoreFunc.func_name]]
    sortedIndices = [i[0] for i in sorted(enumerate(pvals), key=lambda x:x[1])]
    return sortedIndices

class SupervisedStdScaler(StandardScaler):
    '''
    A standard scaler that uses group labels to Scale
    '''

    def __init__(self):
        self.__subscaler = StandardScaler()
        self.__subscaleru = StandardScaler()

    def fit(self, X, y=None, label=None):
        if not (y is None or label is None):
            x_used = X[y == label]
        else:
            x_used = X
        self.__subscaler.fit(x_used)
        self.__subscaleru.fit(X)

    def transform(self, X, y=None, label=None):
        scale = self.__subscaler.scale_
        center = self.__subscaleru.mean_
        tile_center = np.tile(center,[X.shape[0], 1])
        tile_scale = np.tile(scale,[X.shape[0], 1])
        X_transf = (X-tile_center) / tile_scale
        return X_transf

class SupervisedRobustScaler(StandardScaler):
    '''
    A standard scaler that uses group labels to Scale
    '''

    def __init__(self):
        self.__subscaler = RobustScaler()
        self.__subscaleru = RobustScaler()

    def fit(self, X, y=None, label=None):
        if not (y is None or label is None):
            x_used = X[y == label]
        else:
            x_used = X
        self.__subscaler.fit(x_used)
        self.__subscaleru.fit(X)

    def transform(self, X, y=None, label=None):
        scale = self.__subscaler.scale_
        center = self.__subscaleru.center_
        tile_center = np.tile(center,[X.shape[0], 1])
        tile_scale = np.tile(scale,[X.shape[0], 1])
        X_transf = (4./3) * (X-tile_center) / tile_scale
        return X_transf
    #If data is normally distributed, it will have unit variance


class Ranker(object):
    """
    Class version of univariate ranking, to pass to multicore jobs
    Inputs:
        data: the full data matrix
        labels: full class labels
        ranking function: the ranking function to give to rank_vars
        rank_vars: the rank_vars function
        fp: list of fold train-test pairs
    """
    def __init__(self, data, labels, ranking_function, rank_vars=rank_vars):
        self.data = data
        self.labels = labels
        self.rf = ranking_function
        self.rank_vars = rank_vars
    def __call__(self, fp):
        rv = self.rank_vars(self.data[fp[0], :], self.labels[fp[0]],
                            self.rf)
        return rv

def get_rank_per_fold(data, labels, fold_pairs, ranking_function=ttest_ind,
                      save_path=None,load_file=True,
                      parallel=True):
    '''
    Applies rank_vars to each test set in list of fold pairs
    Inputs:
        data: array
            features for all samples
        labels: array
            label vector of each sample
        fold_pair: list
            list pairs of index arrays containing train and test sets
        ranking_function: function object, default: ttest_ind
            function to apply for ranking features
        ranking_function: function
            ranking function to use, default: ttest_ind
        save_path: dir to load and save ranking files
        load_file: bool
            Whether to try to load an existing file, default: True
        parallel: bool
            True if multicore processing is desired, default: True        
    Outputs:
        rank_per_fod: list
            List of ranked feature indexes for each fold pair
    '''
    file_loaded = False
    if load_file:
        if isinstance(save_path, str):            
            fname = path.join(save_path, "{}_{}_folds.mat".format(
                              ranking_function.__name__, len(fold_pairs)))
            try: 
                rd = scipy.io.loadmat(fname, mat_dtype = True)
                rank_per_fold = rd['rank_per_fold']
                file_loaded = True
            except:
                pass
        else:
            print('No rank file path: Computing from scratch without saving')
    if not file_loaded:        
        if not parallel:
            rank_per_fold = []
            for fold_pair in fold_pairs:
                rankedVars = rank_vars(data[fold_pair[0], :],
                                       labels[fold_pair[0]], ranking_function)
                rank_per_fold.append(rankedVars)
        else:
            pool = Pool(processes=min(len(fold_pairs), PROCESSORS))
            rank_per_fold = pool.map(Ranker(data, labels, ranking_function,
                                            rank_vars),fold_pairs)
            pool.close()
            pool.join()
        if isinstance(save_path, str):
            fname = path.join(save_path, "{}_{}_folds.mat".format(
                              ranking_function.__name__, len(fold_pairs)))
            with open(fname, 'wb') as f:
                scipy.io.savemat(f, {'rank_per_fold': rank_per_fold})
    return rank_per_fold

def make_classifiers(NAMES) :
    """Function that makes classifiers each with a number of folds.

    Returns two dictionaries for the classifiers and their parameters, using
    `data_shape` and `ksplit` in construction of classifiers.

    Parameters
    ----------
    data_shape : tuple of int
        Shape of the data.  Must be a pair of integers.
    ksplit : int
        Number of folds.

    Returns
    -------
    classifiers: dict
        The dictionary of classifiers to be used.
    params: dict
        A dictionary of list of dictionaries of the corresponding
        params for each classifier.
    """

#    if len(data_shape) != 2:
#        raise ValueError("Only 2-d data allowed (samples by dimension).")

    classifiers = {
        "Chance": DummyClassifier(strategy="most_frequent"),
        "Nearest Neighbors": KNeighborsClassifier(3),
        "Linear SVM": LinearSVC(penalty='l2', C=1,# probability=True,
                          class_weight='balanced'),
        # sahil changed the configuration from "probability" True to False (probability
        #  based inference doesn't work well in SVM models from part experiences,
        # as SVM original algorithm just split the data with no probablistic notion of inference.)
        "RBF SVM": SVC(gamma=2, C=1, probability=False),
        "Decision Tree": DecisionTreeClassifier(max_depth=None,
                                                max_features="auto"),
        "Random Forest": RandomForestClassifier(max_depth=None,
                                                n_estimators=20,
                                                max_features="auto",
                                                n_jobs=PROCESSORS),
        "Logistic Regression": LogisticRegression(penalty='l1',
                                                   class_weight='balanced'),
        "Naive Bayes": GaussianNB(),
        "LDA": LDA(),
        "SGD_logL1": SGDClassifier(random_state=1952,loss='log', average = 3,
                                  penalty='l1',
                                  alpha=1e-3,
                                  class_weight='balanced'),
        "SGD_log_elastic": SGDClassifier(random_state=1952,loss='log',
                                          class_weight='balanced',
                                          alpha=1e-3,
                                          average = 3,
                                          penalty='elasticnet'),
        "SGD_SVM_elastic": SGDClassifier(random_state=1952,loss='log',
                                          class_weight='balanced',
                                          average = 3,
                                          alpha=1e-3,
                                          penalty='elasticnet'),

        # Sahil commented the two classiifer below as not able to install the packages
        # "CGC_log_L1": CDClassifier(penalty="l1",
        #            loss="log",
        #            multiclass=False,
        #            max_iter=200,
        #            C=1,
        #            tol=1e-3),
        # "SDCA_SVM_elastic": SDCAClassifier(
        #            loss="hinge",
        #            max_iter=200,
        #            tol=1e-3)
        #
        }

    params = {
        "Chance": {},
        #  3, 4, 5
        # , 6, 7, 8, 10, 12, 15, 20, 30, 50, 75, 100
        #
        #
        "Nearest Neighbors": {"n_neighbors": [1, 2, 3, 5, 10, 20, 50, 75, 100, 150, 200, 250]},
        "Linear SVM": {"C": [0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2],
                       "loss":['hinge', 'squared_hinge']},
        "RBF SVM": {"kernel": ["rbf"],
                     "gamma": np.logspace(-2, 0, 6).tolist() + \
                              np.logspace(0,1,5)[1:].tolist(),
                    "C": np.logspace(-2, 2, 5).tolist()},
                    "Decision Tree": {},
        "Random Forest": {"max_depth": np.round(np.logspace(np.log10(2), \
                                       1.2, 6)).astype(int).tolist()},
        "Logistic Regression": {"C": np.logspace(-2, 3, 6).tolist()},
        "Naive Bayes": {},
        "LDA": {},
        "SGD_logL1": {"alpha": np.logspace(-5, 2, 7)},
        "SGD_log_elastic": {"alpha": np.logspace(-5, 2, 6),
                            "l1_ratio": 10**np.array([-2, -1, -.5, -.25,
                                                      -.12, -.06, -.01])},
        "SGD_SVM_elastic": {"alpha": np.logspace(-5, 2, 6),
                            "l1_ratio": 10**np.array([-2, -1, -.5, -.25,
                                                      -.12, -.06, -.01])},
        "CGC_log_L1": {"alpha": np.logspace(-5, 2, 6)},
        "SDCA_SVM_elastic": {"alpha": np.logspace(-4, 4, 5),
                             "l1_ratio": 10**np.array([-3,-2, -1, np.log10(.5),
                                                       np.log10(.9)])}
            }
    out_classifiers = {cname: classifiers[cname] for cname in NAMES}
    out_params = {cname: params[cname] for cname in NAMES}
    logging.info("Using classifiers %r with params %r" % (out_classifiers,
                                                         out_params))
    return classifiers, params


class per_split_classifier(object):
    """
    Class version of classify function, to pass to multicore jobs
    Inputs:
        data: the full data matrix
        labels: full class labels
        classifier: classifier object to use
        numTopVars: list of top variables to use
        zipped_ranks_n_fp: zipped list 2-tuple with ranked vars and train-test
                           indices
        fp: a single train-test pair
    """
    def __init__(self, data, labels, classifier, numTopVars):
        self.data = data
        self.labels = labels
        self.clf = classifier
        self.numTopVars = numTopVars

    def __call__(self, zipped_ranks_n_fp):
        rankedVars, fp = zipped_ranks_n_fp
        confMats = []
        totalErrs = []
        fitted_classifiers = []
        for numVars in self.numTopVars:
            confMat, totalErr, fitted_classifier = classify(self.data[:, rankedVars[:numVars]],
                                                            self.labels, fp, self.clf)
            confMats.append(confMat)
            totalErrs.append(totalErr)
            fitted_classifiers.append(fitted_classifier)
        return confMats, totalErrs, fitted_classifiers


def get_score(data, labels, fold_pairs, name, model, param, numTopVars,
              rank_per_fold=None, parallel=True, rand_iter=-1):
    """
    Function to get score for a classifier.

    Parameters
    ----------
    data: array_like
        Data from which to derive score.
    labels: array_like or list
        Corresponding labels for each sample.
    fold_pairs: list of pairs of array_like
        A list of train/test indicies for each fold
        dhjelm(Why can't we just use the KFold object?)
    name: str
        Name of classifier.
    model: WRITEME
    param: WRITEME
        Parameters for the classifier.
    parallel: bool
        Whether to run folds in parallel. Default: True

    Returns
    -------
    classifier: WRITEME
    allConfMats: Confusion matrix for all folds and all variables sets and best performing parameter set
                 ([numFolds, numVarSets]) 
    """
    assert isinstance(name, str)
    logging.info("Classifying %s" % name)
    ksplit = len(fold_pairs)
#    if name not in NAMES:
#        raise ValueError("Classifier %s not supported. "
#                         "Did you enter it properly?" % name)

    # Redefine the parameters to be used for RBF SVM (dependent on
    # training data)
    if "SGD" in name:
        param["n_iter"] = [25]  # [np.ceil(10**3 / len(fold_pairs[0][0]))]
    classifier = get_classifier(name, model, param, rand_iter=rand_iter)
                    
    if name == "RBF SVM": #This doesn't use labels, but looks as ALL data
        logging.info("RBF SVM requires some preprocessing."
                    "This may take a while")
        #
        is_data_computed_gamma = True
        #
        if not is_data_computed_gamma:
            # Sahil commented the code below that computes the gamma choices from data.
            # The computed gamma choices seem too low thereby making SVM very slow. Instead, trying out fixed values.
            print param
            gamma = param['gamma']
            gamma = np.array(gamma)
            print 'gamma', gamma
        else:
            #Euclidean distances between samples
            # sahil switched from the first call to second one for computing the dist as the first one is giving error.
            # dist = pdist(StandardScaler().fit(data), "euclidean").ravel()
            dist = pdist(RobustScaler().fit_transform(data), "euclidean").ravel()
            print 'dist', dist
            #Estimates for sigma (10th, 50th and 90th percentile)
            sigest = np.asarray(np.percentile(dist, [10, 50, 90]))
            print 'sigest', sigest
            #Estimates for gamma (= -1/(2*sigma^2))
            gamma = 1./(2*sigest**2)
            print 'gamma', gamma
        #
        #
        #Set SVM parameters with these values
        # sahil changed the code a bit to remove a bug
        # param = [{"kernel": ["rbf"],
        #           "gamma": gamma.tolist(),
        #           "C": np.logspace(-2,2,5).tolist()}]
        param = {"kernel": ["rbf"],
                  "gamma": gamma.tolist(),
                  "C": np.logspace(-2, 2, 5).tolist()}
    # if name not in ["Decision Tree", "Naive Bayes"]:
    if param:
        if hasattr(classifier,'param_grid'): 
        # isinstance(classifier, GridSearchCV):
            print 'param', param
            N_p = np.prod([len(l) for l in param.values()])
        elif isinstance(classifier, RandomizedSearchCV):
            N_p = classifier.n_iter
    else:
        N_p = 1
#    is_cv = isinstance(classifier, GridSearchCV) or \
#            isinstance(classifier, RandomizedSearchCV)
#    print('Name: {}, ksplit: {}, N_p: {}'.format(name, ksplit, N_p))
    if (not parallel) or ksplit <= N_p or \
    (name == "Random Forest") or ("SGD" in name):
        logging.info("Attempting to use grid search...")
        classifier.n_jobs = PROCESSORS
        classifier.pre_dispatch = 1 # np.floor(PROCESSORS/24)
        allConfMats = []
        allTotalErrs = []
        allFittedClassifiers = []
        for i, fold_pair in enumerate(fold_pairs):
            confMats = []
            totalErrs = []
            fitted_classifiers = []
            logging.info("Classifying a %s the %d-th out of %d folds..."
                   % (name, i+1, len(fold_pairs)))
            if rank_per_fold is not None:
                rankedVars = rank_per_fold[i]
            else:
                rankedVars = np.arange(data.shape[1])
            #
            for numVars in numTopVars:
                logging.info('Classifying for top %i variables' % numVars)
                #
                # print 'rankedVars', rankedVars
                #
                confMat, totalErr, fitted_classifier = classify(data[:, rankedVars[:numVars]],
                                                                labels,
                                                                fold_pair,
                                                                classifier)
                confMats.append(confMat)
                totalErrs.append(totalErr)
                fitted_classifiers.append(fitted_classifier)
            # recheck the structure of area and fScore variables
            allConfMats.append(confMats)
            allTotalErrs.append(totalErrs)
            allFittedClassifiers.append(fitted_classifiers)
    else:
        print 'parallel computing going on (debug Sahil ...) ..........................'
        #
        classifier.n_jobs = PROCESSORS
        logging.info("Multiprocessing folds for classifier {}.".format(name))
        pool = Pool(processes=min(ksplit, PROCESSORS))
        out_list = pool.map(per_split_classifier(data, labels, classifier,
                                                 numTopVars),
                            zip(rank_per_fold, fold_pairs))
        pool.close()
        pool.join()
        #allConfMats = [el[0] for el in out_list]
        #allTotalErrs = [el[1] for el in out_list]
        #allFittedClassifiers = [el[2] for el in out_list]
        allConfMats, allTotalErrs, allFittedClassifiers = tuple(zip(*out_list))
    return classifier, allConfMats, allTotalErrs, allFittedClassifiers

def get_classifier(name, model, param, rand_iter=-1):
    """
    Returns the classifier for the model.

    Parameters
    ----------
    name: str
        Classifier name.
    model: WRITEME
    param: WRITEME
    data: array_like, optional

    Returns
    -------
    WRITEME
    """
    assert isinstance(name, str)
    if param: # Do grid search only if parameter list is not empty
        N_p = np.prod([len(l) for l in param.values()])
        if (N_p <= rand_iter) or rand_iter<=0:
            logging.info("Using grid search for %s" % name)
            model = GridSearchCV(model, param, cv=5, scoring="accuracy",
                                 n_jobs=PROCESSORS)
        else:
            logging.info("Using random search for %s" % name)
            model = RandomizedSearchCV(model, param, cv=5, scoring="accuracy",
                                 n_jobs=PROCESSORS, n_iter=rand_iter)
    else:
        logging.info("Not using grid search for %s" % name)
    return model

def classify(data, labels, (train_idx, test_idx), classifier=None):

    """
    Classifies given a fold and a model.

    Parameters
    ----------
    data: array_like
        2d matrix of observations vs variables
    labels: list or array_like
        1d vector of labels for each data observation
    (train_idx, test_idx) : list
        set of indices for splitting data into train and test
    classifier: sklearn classifier object
        initialized classifier with "fit" and "predict_proba" methods.
    Returns
    -------
    WRITEME
    """

    assert classifier is not None, "Why would you pass not classifier?"

    # Data scaling based on training set
    scaler = SupervisedStdScaler() #SupervisedRobustScaler()  #  # 
    scaler.fit(data[train_idx,:], labels[train_idx], label=-1)
    #scaler.fit(data[train_idx,:], labels[train_idx])
    data_train = scaler.transform(data[train_idx,:])
    data_test = scaler.transform(data[test_idx,:])
    try:
        classifier.fit(data_train, labels[train_idx])
    
        
        confMat = confusion_matrix(labels[test_idx],
                                  classifier.predict(data_test))
        if confMat.shape == (1,1):
            if all(labels[test_idx] == -1):
                confMat = np.array([[confMat[0], 0], [0, 0]], dtype=confMat.dtype)
            else:
                confMat = np.array([[0, 0], [0, confMat[0]]], dtype=confMat.dtype)
        confMatRate = confMat / np.tile(np.sum(confMat, axis=1).astype('float'), (2,1)).transpose()
        totalErr = (confMat[0, 1] + confMat[1, 0]) / float(confMat.sum())
        #if type(classifier) not in [type(None), DummyClassifier]:
        if hasattr(classifier,'param_grid'): 
        #isinstance(classifier, GridSearchCV) or \
        #    isinstance(classifier, RandomizedSearchCV):
                fitted_model = classifier.best_estimator_
        else:
                fitted_model = copy.copy(classifier)        
        return confMatRate, totalErr, fitted_model
    except np.linalg.linalg.LinAlgError as e:
        # sahil added statement to raise the error instead of returning nun values
        print e.message
        raise e
        # return np.array([[np.nan, np.nan], [np.nan, np.nan]]), np.nan, None


def load_data(data_file, data_pattern='*.mat'):
    """
    Loads the data from multiple sources if provided.

    Parameters
    ----------
    data_file: str
    data_pattern: str

    Returns
    -------
    data: array_like
    """
    
    dataMat = scipy.io.loadmat(data_file, mat_dtype=True)
    data = dataMat['data']

    logging.info("Data loading complete. Shape is %r" % (data.shape,))
    return data[:, :-1], data[:, -1], data_file

def load_labels(source_dir, label_pattern):
    """
    Function to load labels file.

    Parameters
    ----------
    source_dir: str
        Source directory of labels
    label_pattern: str
        unix regex for label files.

    Returns
    -------
    labels: array_like
        A numpy vector of the labels.
    """

    logging.info("Loading labels from %s with pattern %s"
                % (source_dir, label_pattern))
    label_files = glob(path.join(source_dir, label_pattern))
    if len(label_files) == 0:
        raise ValueError("No label files found with pattern %s"
                         % label_pattern)
    if len(label_files) > 1:
        raise ValueError("Only one label file supported ATM.")
    labels = np.load(label_files[0]).flatten()
    logging.info("Label loading complete. Shape is %r" % (labels.shape,))
    return labels

def save_classifier_results(classifier_name, out_dir, allConfMats, allTotalErrs):
    """saves the classifier results including TN, FN and total error. Plot FP and FN.
    """

    # convert confusion matrix and total errors into numpy array
    tmpAllConfMats = np.array(allConfMats)
    tmpAllTotalErrs = np.array(allTotalErrs)
    # initialize mean and std variables
    TN_means = np.zeros(tmpAllConfMats.shape[1])
    TN_stds = np.zeros(tmpAllConfMats.shape[1])
    FN_means = np.zeros(tmpAllConfMats.shape[1])
    FN_stds = np.zeros(tmpAllConfMats.shape[1])
    total_means = np.zeros(tmpAllConfMats.shape[1])
    total_stds = np.zeros(tmpAllConfMats.shape[1])

    for j in range(tmpAllConfMats.shape[1]):
        tmpData = tmpAllConfMats[:, j, 0, 0]
        TN_means[j] = np.mean(tmpData[np.invert(np.isnan(tmpData))])
        TN_stds[j] = np.std(tmpData[np.invert(np.isnan(tmpData))])
        tmpData = tmpAllConfMats[:, j, 1, 0]
        FN_means[j] = np.mean(tmpData[np.invert(np.isnan(tmpData))])
        FN_stds[j] = np.std(tmpData[np.invert(np.isnan(tmpData))])
        tmpData = tmpAllTotalErrs[:, j]
        # Compute mean of std of non-Nan values
        total_means[j] = np.mean(tmpData[np.invert(np.isnan(tmpData))])
        total_stds[j] = np.std(tmpData[np.invert(np.isnan(tmpData))])
    with open(path.join(out_dir, classifier_name+'_errors.mat'),'wb') as f:
        scipy.io.savemat(f, {'TN_means': TN_means,
                             'TN_stds': TN_stds,
                             'FN_means': FN_means,
                             'FN_stds': FN_stds,
                             'total_means': total_means,
                             'total_stds': total_stds,
                             })


def save_classifier_object(clf, FittedClassifiers, name, out_dir):
    if out_dir is not None:
        save_path = path.join(out_dir, name + '.pkl')
        logging.info("Saving classifier to %s" % save_path)
        classifier_dict = {'name': name,
                           'classifier': clf,
                           'FittedClassifiers': FittedClassifiers}
        with open(save_path, "wb") as f:
            pickle.dump(classifier_dict, f)


def save_combined_results(NAMES, dscore, totalErrs, numTopVars, out_dir, filebase):
    confMatResults = {name.replace(" ", ""): scores for name, scores in zip(NAMES, dscore)}
    confMatResults['topVarNumbers'] = numTopVars
    totalErrResults = {name.replace(" ", ""): errs for name, errs in zip(NAMES, totalErrs)}
    totalErrResults['topVarNumbers'] = numTopVars
    # save results from all folds
    # dscore is a matrix [classifiers, folds, #vars, 2, 2]
    dscore = np.asarray(dscore)
    totalErrs = np.asarray(totalErrs)
    with open(path.join(out_dir, filebase + '_dscore_array.mat'), 'wb') as f:
        scipy.io.savemat(f, {'dscore': dscore,
                             'topVarNumbers': numTopVars,
                             'classifierNames': NAMES})

    with open(path.join(out_dir, filebase + '_errors_array.mat'), 'wb') as f:
        scipy.io.savemat(f, {'errors': totalErrs,
                             'topVarNumbers': numTopVars,
                             'classifierNames': NAMES})
    # Save all results
    with open(path.join(out_dir, 'confMats.mat'),'wb') as f:
        scipy.io.savemat(f, confMatResults)
    with open(path.join(out_dir, 'totalErrs.mat'),'wb') as f:
        scipy.io.savemat(f, totalErrResults)


def plot_errors(NAMES, numTopVars, dscore=None, totalErrs=None,
                filename_base='', out_dir=None, compute_error=True,
                format_used='png', fmt_str=None):
    #
    # Sahil added
    assert fmt_str is not None
    # Sahil added
    assert NAMES is not None and NAMES
    error_plots_map = {}
    error_plots_map['numTopVars'] = numTopVars
    # error_plots_map['classifier_names'] = NAMES
    for curr_name in NAMES:
        error_plots_map[curr_name] = {}
    #
    ######################################
    # Plot Figures
    # Confusion matrix format is:
    #   TN  FP
    #   FN  TP
    # Plotting false-positive ratio
    cl = [(1., 0., 0.),
          (0., 1., 0.),
          (0., 0., 1.),
          (0., 0., 0.),
          (.5, .5, 0.),
          (.5, 0., .5),
          (0., .5, .5),
          (.9, .9, .1),
          (0., 1., 1.),
          (1., 0., 1.),
          (1., .7, .3),
          (.5, 1., .7),
          (.7, .3, 1.),
          (.3, .7, 1.),
          (.3, .1, .7),
          (1., .3, .7)]

    ax = pl.gca()
#       ax.set_prop_cycle(color=cl)

    if not compute_error:
        dscore = range(len(NAMES))
        totalErrs = range(len(NAMES))
    if dscore:
        dscore = np.asarray(dscore)
        #
        # Plotting FP rate
        handles = []
        means = np.zeros(len(numTopVars))  # np.zeros(dscore.shape[2])
        # sahil added code line for stds
        stds = np.zeros(dscore.shape[2])
        #
        for i in range(dscore.shape[0]):
            if compute_error:
                for j in range(dscore.shape[2]):
                    tmpData = dscore[i, :, j, 0, 1]
                    # Compute mean of std of non-Nan values
                    means[j] = np.mean(tmpData[np.invert(np.isnan(tmpData))])
                    # # sahil uncommented this lin below
                    stds[j] = np.std(tmpData[np.invert(np.isnan(tmpData))])
            else: 
                name=NAMES[i]
                fn = path.join(out_dir, 'confMats.mat')
                dm = scipy.io.loadmat(fn)
                z=dm[name]
                xfold_mean = np.nanmean(z, 0)
                result_file = path.join(out_dir, name + '_errors.mat')
                err_dict = scipy.io.loadmat(result_file, mat_dtype=True)
                means = xfold_mean[:, 0, 1]
                #
            #
            # Sahil added
            error_plots_map[NAMES[i]]['false_positives'] = {}
            error_plots_map[NAMES[i]]['false_positives']['means'] = means.tolist()
            error_plots_map[NAMES[i]]['false_positives']['stds'] = stds.tolist()
            #
            handles.append(pl.errorbar(numTopVars, means, stds, fmt=fmt_str))
            #
        ax.set_title(filename_base, fontsize=FONT_SIZE-4)
        ax.set_xscale('log')
        ax.set_ylabel('FP rate', fontsize=FONT_SIZE)
        ax.set_xlabel('Number of top variables', fontsize=FONT_SIZE)
        ax.set_xlim(left = min(numTopVars)-1, right=max(numTopVars) + 100)
        # ax.set_ylim((0,1))
        # ax.set_yticks(np.arange(0, 1.1, 0.1))
        pl.legend(handles, NAMES, bbox_to_anchor=(1, 1), loc=2,
                  borderaxespad=0., prop={'size':14})
        pl.grid()
        
        if out_dir is not None:
            # change the file you're saving it to
            pl.savefig(path.join(out_dir, filename_base + '_FP.' + format_used),
                       dpi=300, bbox_inches='tight', format=format_used)
        else:
            pl.show(True)
        pl.cla()
    
    
        handles = []
        # Plotting false-negative ratio
        means = np.zeros(len(numTopVars))
        stds = np.zeros(dscore.shape[2])
      #  ax.set_prop_cycle(color=cl)
        for i in range(dscore.shape[0]):
            if compute_error:
                for j in range(dscore.shape[2]):
                    tmpData = dscore[i, :, j, 1, 0]
                    # Compute mean of std of non-Nan values
                    means[j] = np.mean(tmpData[np.invert(np.isnan(tmpData))])
                    stds[j] = np.std(tmpData[np.invert(np.isnan(tmpData))])
            else:
                name = NAMES[i]
                result_file = path.join(out_dir, name + '_errors.mat')
                err_dict = scipy.io.loadmat(result_file, mat_dtype=True)
                means = err_dict['FN_means'].flatten()
                stds = err_dict['FN_stds'].flatten()
            #
            # Sahil added
            error_plots_map[NAMES[i]]['false_negatives'] = {}
            error_plots_map[NAMES[i]]['false_negatives']['means'] = means.tolist()
            error_plots_map[NAMES[i]]['false_negatives']['stds'] = stds.tolist()
            #
            handles.append(pl.errorbar(numTopVars, means, stds, fmt=fmt_str))
        #
        ax = pl.gca()
        ax.set_xscale('log')
        ax.set_ylabel('FN rate', fontsize=FONT_SIZE)
        ax.set_xlabel('Number of top variables', fontsize=FONT_SIZE)
        ax.set_title(filename_base, fontsize=FONT_SIZE-4)
        ax.set_xlim(left = min(numTopVars)-1, right=max(numTopVars) + 100)
        # ax.set_ylim((0,1))
        # ax.set_yticks(np.arange(0, 1.1, 0.1))
        pl.legend(handles, NAMES, bbox_to_anchor=(1, 1), loc=2,
                  borderaxespad=0.,prop={'size':14})
        pl.grid()
    
        if out_dir is not None:
            # change the file you're saving it to
            pl.savefig(path.join(out_dir, filename_base + '_FN.' + format_used),
                       dpi=300, bbox_inches='tight', format=format_used)
        else:
            pl.show(True)
        pl.cla()
    #
    if totalErrs:
        totalErrs = np.asarray(totalErrs)
        handles = []
        # Plotting total error
        means = np.zeros(len(numTopVars))
        stds = np.zeros(totalErrs.shape[2])
        ax = pl.gca()
     #   ax.set_prop_cycle(color=cl)
        for i in range(totalErrs.shape[0]):
            if compute_error:
                for j in range(totalErrs.shape[2]):
                    tmpData = totalErrs[i, :, j]
                    # Compute mean of std of non-Nan values
                    means[j] = np.mean(tmpData[np.invert(np.isnan(tmpData))])
                    stds[j] = np.std(tmpData[np.invert(np.isnan(tmpData))])
            else:
                name = NAMES[i]
                result_file = path.join(out_dir, name + '_errors.mat')
                err_dict = scipy.io.loadmat(result_file, mat_dtype=True)
                means = err_dict['total_means'].flatten()
                stds = err_dict['total_stds'].flatten()
            #
            # Sahil added
            error_plots_map[NAMES[i]]['total_error'] = {}
            error_plots_map[NAMES[i]]['total_error']['means'] = means.tolist()
            error_plots_map[NAMES[i]]['total_error']['stds'] = stds.tolist()
            #
            handles.append(pl.errorbar(numTopVars, means, stds, fmt=fmt_str))
        #
        ax = pl.gca()
        ax.set_title(filename_base, fontsize=FONT_SIZE-4)
        ax.set_xscale('log')
        ax.set_axis_bgcolor('w')
        ax.set_xlabel('Number of top variables', fontsize=FONT_SIZE)
        ax.set_ylabel('Error rate', fontsize=FONT_SIZE)
        ax.set_xlim(left = min(numTopVars)-1, right=max(numTopVars) + 100)
        # ax.set_ylim((0,1))
        # ax.set_yticks(np.arange(0, 1.1, 0.1))
        pl.legend(handles, NAMES, bbox_to_anchor=(1, 1), loc=2,
                  borderaxespad=0., prop={'size':14})
        pl.grid()
    
        if out_dir is not None:
            # change the file you're saving it to
            pl.savefig(path.join(out_dir, filename_base + '_total_errors.' + format_used),
                       dpi=300, bbox_inches='tight', format=format_used)
        else:
            pl.show(True)
    pl.clf()
    #
    #
    with open(path.join(out_dir, filename_base + '_plots_data.json'), 'w') as f_json:
        json.dump(error_plots_map, f_json, indent=4, sort_keys=True)

def main(source_dir, ksplit, out_dir, data_pattern, label_pattern, test_mode,
         numTopVars = [10, 50, 100, 500, 1000], compute_results=True):
    """
    Main function for polyssifier.

    Parameters
    ----------
    source_dir: str
    ksplit: int
    out_dir: str
    data_pattern: str
        POSIX-type regex string for list of paths.
    label_pattern: str
        POSIX-type regex string for list of paths.
    test_mode: bool
    """
    # Load input and labels.
    data, labels, data_file = load_data(source_dir, data_pattern)
    filename_base = path.splitext(path.basename(mname))[0]
    # Get classifiers and params.
    global NAMES
    if test_mode:
        NAMES = ["Chance", "Nearest Neighbors", "Linear SVM", "Decision Tree",
                 "Logistic Regression", "Naive Bayes", "LDA"]
        ksplit = 3

    classifiers, params = make_classifiers(NAMES)  # data.shape, ksplit)


    # Make the folds.
    logging.info("Making %d folds" % ksplit)
    #kf = StratifiedKFold(labels, n_folds=ksplit)
    kf = KFold(labels.shape[0], n_folds=ksplit)

    # Extract the training and testing indices from the k-fold object,
    # which stores fold pairs of indices.
    fold_pairs = [(tr, ts) for (tr, ts) in kf]
    assert len(fold_pairs) == ksplit
    rank_per_fold = get_rank_per_fold(data, labels, fold_pairs,
                                      save_path=out_dir, parallel=True)
    #dhjelm: were we planning on using this dict?
    #score={}
    dscore=[]
    totalErrs = []
    if compute_results:
        for name in NAMES:
            mdl = classifiers[name]
            param = params[name]
            # get_score runs the classifier on each fold,
            # each subset of selected top variables and does a grid search for
            # classifier-specific parameters (selects the best)
            clf, allConfMats, allTotalErrs, allFittedClassifiers = \
                get_score(data, labels, fold_pairs, name, mdl, param,
                          numTopVars=numTopVars,
                          rank_per_fold=rank_per_fold, parallel=True,
                          rand_iter=-1)
            # save classifier object and results to file
            save_classifier_results(name, out_dir, allConfMats,
                                    allTotalErrs)
            save_classifier_object(clf, allFittedClassifiers, name, out_dir)
            # Append classifier results to list of all results
            dscore.append(allConfMats)
            totalErrs.append(allTotalErrs)
        '''
        First do some saving of total results
        '''
        save_combined_results(NAMES, dscore, totalErrs,
                              numTopVars, out_dir, filename_base)

    plot_errors(NAMES, numTopVars, dscore, totalErrs,
                filename_base, out_dir,compute_results)

    logging.shutdown()



def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory",
                        help="Directory where the data files live.")
    parser.add_argument("out", help="Output directory of files.")
    parser.add_argument("-t", "--test", action="store_true",
                        help=("Test mode, avoids slow classifiers and uses"
                              " 3 folds"))
    parser.add_argument("--folds", default=10,
                        help="Number of folds for n-fold cross validation")
    parser.add_argument("--data_pattern", default="*.mat",
                        help="Pattern for data files")
    parser.add_argument("--label_pattern", default="*.mat",
                        help="Pattern for label files")
    return parser

if __name__ == "__main__":
    CPUS = multiprocessing.cpu_count()
    if CPUS < PROCESSORS:
        raise ValueError("Number of PROCESSORS exceed available CPUs, "
                         "please edit this in the script and come again!")
    
    # numTopVars = [50, 100, 300, 900, 2700]
    #numTopVars = [10, 50]

    parser = make_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s",
                        filename=path.join(args.out, 'log.log'),
                        filemode='w',
                        level=logging.DEBUG)
    logger = logging.getLogger()
    ch = logging.StreamHandler(logging.sys.stdout)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    main(args.data_directory, out_dir=args.out, ksplit=int(args.folds),
         data_pattern=args.data_pattern, label_pattern=args.label_pattern,
         test_mode=args.test, numTopVars=numTopVars)

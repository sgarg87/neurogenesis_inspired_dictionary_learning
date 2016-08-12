# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:33:24 2016

@author: pipolose
"""


import polyssifier_21_4_16 as ps
import logging
from sklearn.cross_validation import KFold
from os import path
import numpy as np
# import ipdb


if __name__ == "__main__":
    #
    #
    import sys
    mname = sys.argv[1]
    out_dir = sys.argv[2]
    # mname = ('./large_image_sparse_codings/mairal_sparse_codings_test.mat')
    # out_dir = './classifier_out'
    #
    #
    '''
    INPUT PARAMETERS
    '''
    #
    ksplit = 5
    # ksplit = 3
    #
    is_all_variables_include = True
    #
    numTopVars = [1, 5, 25, 50, 100, 200, 350, 500, 750]
    # numTopVars = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 35, 50, 65, 75, 85, 100, 125, 150, 175, 200, 225]
    # numTopVars = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 35, 50, 65, 75, 85, 100, 125, 150, 175]
    # numTopVars = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 35, 50, 65, 75, 85, 100, 125]
    # numTopVars = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 35, 50, 65, 75, 85]
    # numTopVars = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25]
    # numTopVars = [5, 10, 25]
    # numTopVars = [10, 30, 100, 300, 1000, 3000, 10000]
    #
    # If false, load classifier results instead of running:
    compute_results = True
    #
    # NAMES = ["RBF SVM", "Logistic Regression", "Naive Bayes", "Nearest Neighbors", "Random Forest"]
    NAMES = ["Logistic Regression", "Naive Bayes", "Nearest Neighbors", "Random Forest"]
    #
    '''
    Initializing logger to write to file and stdout
    '''
    logging.basicConfig(format="[%(asctime)s:%(module)s:%(levelname)s]:%(message)s",
                        filename=path.join(out_dir, 'log.log'),
                        filemode='w',
                        level=logging.DEBUG)
    logger = logging.getLogger()
    ch = logging.StreamHandler(logging.sys.stdout)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    '''
    DATA LOADING
    '''
    data, labels, data_file = ps.load_data(mname)
    print 'data.dtype', data.dtype
    if np.any(np.isnan(data)):
        h = np.nonzero(np.isnan(data))
        data[h[0], h[1]] = 0
        logging.warning('Nan values were removed from data')
    if np.any(np.isinf(data)):
        h = np.nonzero(np.isinf(data))
        data[h[0], h[1]] = 0
        logging.warning('Inf values were removed from data')
    #
    #
    #
    # use all the variables also.
    if is_all_variables_include:
        numTopVars.append(data.shape[1])
    #
    #
    #
    filename_base = path.splitext(path.basename(mname))[0]
    #
    '''
    CLASSIFIER AND PARAM DICTS
    '''
    classifiers, params = ps.make_classifiers(NAMES)  # data.shape, ksplit)

    kf = KFold(labels.shape[0], n_folds=ksplit)
    # Extract the training and testing indices from the k-fold object,
    # which stores fold pairs of indices.
    fold_pairs = [(tr, ts) for (tr, ts) in kf]
    assert len(fold_pairs) == ksplit

    '''
    RANK VARIABLES FOR EACH FOLD (does ttest unless otherwise specified)
    '''

    rank_per_fold = ps.get_rank_per_fold(data, labels, fold_pairs,
                                         save_path=out_dir, parallel=False, load_file=False)
    print 'rank_per_fold', rank_per_fold
    #
    #
    '''
    COMPUTE SCORES
    '''
    #
    #
    score = {}
    dscore = []
    totalErrs = []
    if compute_results:
        for name in NAMES:
            mdl = classifiers[name]
            param = params[name]
            # get_score runs the classifier on each fold,
            # each subset of selected top variables and does a grid search for
            # classifier-specific parameters (selects the best)
            clf, allConfMats, allTotalErrs, allFittedClassifiers = \
                ps.get_score(data, labels, fold_pairs, name, mdl, param,
                             numTopVars=numTopVars,
                             rank_per_fold=rank_per_fold, parallel=False,
                             rand_iter=-1)
            # save classifier object and results to file
            ps.save_classifier_results(name, out_dir, allConfMats,
                                       allTotalErrs)
            ps.save_classifier_object(clf, allFittedClassifiers, name, out_dir)
            # Append classifier results to list of all results
            dscore.append(allConfMats)
            totalErrs.append(allTotalErrs)

        '''
        First do some saving of total results
        '''
        ps.save_combined_results(NAMES, dscore, totalErrs,
                                 numTopVars, out_dir, filename_base)

    ps.plot_errors(NAMES, numTopVars, dscore, totalErrs,
                   filename_base, out_dir,compute_results,format_used='pdf')

    logging.shutdown()

import numpy as np
from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import time
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from scipy.stats import expon as sp_expon
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_score

import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
#%matplotlib inline

import csv



# an implementation of Kernel Mean Matchin
# referenres:
#  1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4 (2009): 5.
#  2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data." Advances in neural information processing systems. 2006.
def kernel_mean_matching(X, Z, kern='lin', B=1.0, eps=None):
    nx = X.shape[0]
    nz = Z.shape[0]
    if eps == None:
        eps = B/math.sqrt(nz)
    if kern == 'lin':
        K = np.dot(Z, Z.T)
        kappa = np.sum(np.dot(Z, X.T)*float(nz)/float(nx),axis=1)
    elif kern == 'rbf':
        K = compute_rbf(Z,Z)
        kappa = np.sum(compute_rbf(Z,X),axis=1)*float(nz)/float(nx)
    else:
        raise ValueError('unknown kernel')

    K = matrix(K)
    kappa = matrix(kappa)
    G = matrix(np.r_[np.ones((1,nz)), -np.ones((1,nz)), np.eye(nz), -np.eye(nz)])
    h = matrix(np.r_[nz*(1+eps), nz*(eps-1), B*np.ones((nz,)), np.zeros((nz,))])

    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol['x'])
    return coef

def compute_rbf(X, Z, sigma=1.0):
    K = np.zeros((X.shape[0], Z.shape[0]), dtype=float)
    for i, vx in enumerate(X):
        K[i,:] = np.exp(-np.sum((vx-Z)**2, axis=1)/(2.0*sigma))
    return K



# from sklearn import cross_validation
# from sklearn.grid_search import RandomizedSearchCV

# ----------------------------------------------------------------------------
#                          Reading input files
# ----------------------------------------------------------------------------


# target dataset


sqlshare='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshare-legros_v5-mostCorrelatedFeatures.csv'
#sqlshare='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlshare-legros_v5-allFeatures.csv'


datasqlshare=np.genfromtxt(sqlshare, dtype=float, delimiter=';',  encoding='utf8', skip_header=1)
#dataSqlshare=datasqlshare[:,2:-2] # full list of features
#dataSqlshare=datasqlshare[:,-8:-2] #only indexes
#dataSqlshare=datasqlshare[:,2:12] # only intrinsic and relative
#dataSqlshare=datasqlshare[:,-8:-3] #only the most correlated to ground truth (>+/-0.5)
#dataSqlshare=datasqlshare[:,-7:-5] #only jaccard and edit index
#dataSqlshare=datasqlshare[:,-6:-4] #only jaccard and cosine
#dataSqlshare=datasqlshare[:,-6:-2] #only jaccard, cosine, CFI, CTI

#dataSqlshare=datasqlshare[:,2:10] #only the most correlated to GT: NCA, NCP,NCT,Common tables index,Common fragments index, Cosine index,Edit index, Jackard index
dataSqlshare=datasqlshare[:,2:] # most correlated +sessionChange
dataSqlshare=datasqlshare[:,-5:-1] #   sessionChange and indexes


# source dataset

#metricspathfloat='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/dopan-smartBI-concat-mostCorrelatedFeatures.csv'
metricspathfloat='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/dopan-smartBI-concat-mostCorrelatedFeatures-v2.csv'
#metricspathfloat='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/dopan-smartBI-concat-v2-allFeatures.csv'


targettab=np.genfromtxt(metricspathfloat, dtype=float, delimiter=';',  encoding='utf8', skip_header=1)
#targets=targettab[:,-2:-1].ravel()
labels=targettab[:,-1]

datatab=np.genfromtxt(metricspathfloat, dtype=float, delimiter=';',  encoding='utf8', skip_header=1)
#data=datatab[:,3:-4] # full list of features
#data=datatab[:,-10:-4] # only indexes
#data=datatab[:,3:13] # only intrinsic and relative
#data=datatab[:,-10:-5] #only the most correlated to ground truth (>+/-0.5)
#data=datatab[:,-9:-7] #only jaccard and edit index
#data=datatab[:,-8:-6] #only jaccard and cosine
#data=datatab[:,-8:-4] #only jaccard, cosine, CFI, CTI

#data=datatab[:,2:10] #only the most correlated to GT: NCA, NCP,NCT,Common tables index,Common fragments index, Cosine index,Edit index, Jackard index
data=datatab[:,2:-1] #most correlated +sessionChange
data=datatab[:,-6:-2] # only sessionChange and indexes


# text rendering mode
verbose = False      # if True all intermediate results are reported
presentation = True  # if True bloc presentation of results for each sampler



n_iter_search = 50 # for parameter estimation
nb_folds = 10 # for model training
trainPercentage=0.95 # for model testing


# svm_mean_accuracy_scores = []
# rf_mean_accuracy_scores = []
# svm_precision_scores = []
# rf_precision_scores = []
# svm_recall_scores = []
# rf_recall_scores = []

X=data
y=labels.ravel()

# X = complete dataset
# y = complete labels

if verbose: print('Initial data set size: ', len(X))


# ------------------------------------------
#   PREPARING OVER / UNDER-SAMPLING OBJECTS
# ------------------------------------------

#OVERSAMPLING

#choose oversampler below

# oversampling with random
overRd = RandomOverSampler(random_state=0)
# oversampling with smote
overSMOTE = SMOTE()
# oversampling with smote borderline1
overSMOTEb1 = SMOTE(kind='borderline1')
# oversampling with smote borderline2
overSMOTEb2 = SMOTE(kind='borderline2')
# oversampling with smote svm
overSMOTEsvm = SMOTE(kind='svm')
# oversampling with adasyn
overADA = ADASYN()

# UNDERSAMPLING
rus = RandomUnderSampler(random_state=0, ratio='not minority')

#samplers = [overRd, overSMOTE, overSMOTEb1, overSMOTEb2, overSMOTEsvm, overADA, rus]
#samplers = [overRd, overSMOTE, overSMOTEb1,  overSMOTEsvm ]
samplers = [overSMOTE ]

# Main loop to estimate different balancing strategies
# ----------------------------------------------------
for sp in samplers:
    if presentation: print("**************************************************")
    if presentation: print("**************************************************")
    print(sp)
    if presentation: print("--------------------------------------------------")

    # SPLITTING DATASET IN TRAIN and TEST

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-trainPercentage, random_state=0)

    # X_train = 80% of X for parameter estimation and cross-validation model training
    # X_test = 20% of X for final scores

    print('Training set size BEFORE sampling: ', len(X_train))
    print('Test set size: ', len(X_test))

    # for param and model learning
    X_train_b, y_train_b = sp.fit_sample(X_train, y_train)

    # if we don't want to balance
    #X_train_b, y_train_b = X_train, y_train

    print('Training set size AFTER sampling: ', len(X_train_b))
    if presentation: print("--------------------------------------------------")

    # PARAMETER ESTIMATION
    # --------------------
    if verbose: print('parameter estimation')

    # train and test the svm classifier to get a mean accuracy score
    svm_clf = svm.LinearSVC(dual = False, penalty = 'l2', class_weight = 'balanced')

    # specify possible hyperparameters
    param_dist = {"C": sp_expon(scale=1)}

    scoring = {'acc': 'accuracy', 'prec': 'precision', 'rec': 'recall', 'f1': 'f1'}
    # run randomized search
    svm_random_search = RandomizedSearchCV(svm_clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, cv=nb_folds,scoring=scoring, refit='acc')
                                           # scoring=['accuracy', 'precision', 'recall'],
                                           # refit = 'accuracy',

    # if done for parameter learning
    # coefficient for reweighting wrt sqlshare, for transferability purpose
    coef = kernel_mean_matching(dataSqlshare, X_train_b, kern='rbf', B=10)
#    coef = kernel_mean_matching(dataSqlshare, X_train_b)
    coef_flattened=coef.flatten()

    start_time=time.time()
    svm_random_search.fit(X_train_b, y_train_b, sample_weight=coef_flattened)
    #svm_random_search.fit(X_train_b, y_train_b)
    best_C_index = svm_random_search.best_index_
    if verbose:
        print("RandomizedSearchCV took %.2f seconds for %d candidates parameter settings." % ((time.time() - start_time), n_iter_search))
        print('C values tested :',svm_random_search.cv_results_['param_C'].data)
        print('Index of the best C value: ',svm_random_search.best_index_)
        print('Best C value ',svm_random_search.best_estimator_)
        print()
        print('-------------------recall---------------------')
        print('Mean values of the recall for each C value : ',svm_random_search.cv_results_['mean_test_rec'])
        print('Mean value of the recall for the best C value (the one that has the best accuracy value)',svm_random_search.cv_results_['mean_test_rec'][best_C_index])
        print('Standard deviation of the recall for each C value',svm_random_search.cv_results_['std_test_rec'])
        print('Standard deviation of the recall for the best C value (the one that has the best accuracy value)',svm_random_search.cv_results_['std_test_rec'][best_C_index])
        print()
        print('-------------------accuracy---------------------')
        print('Mean values of the accuracy for each C value :',svm_random_search.cv_results_['mean_test_acc'])
        print('Mean value of the accuracy for the best C value',svm_random_search.cv_results_['mean_test_acc'][best_C_index])
        print('Standard deviation of the accuracy for each C value',svm_random_search.cv_results_['std_test_acc'])
        print('Standard deviation of the accuracy for the best C value',svm_random_search.cv_results_['std_test_acc'][best_C_index])
        print()
        print('-------------------precision---------------------')
        print('Mean values of the precision for each C value : ',svm_random_search.cv_results_['mean_test_prec'])
        print('Mean value of the precision for the best C value (the one that has the best accuracy value)',svm_random_search.cv_results_['mean_test_prec'][best_C_index])
        print('Standard deviation of the precision for each C value',svm_random_search.cv_results_['std_test_prec'])
        print('Standard deviation of the precision for the best C value (the one that has the best accuracy value)',svm_random_search.cv_results_['std_test_prec'][best_C_index])
        print()


    best_C = svm_random_search.best_params_["C"]
    if verbose: print("Best value of C found on training set : ", best_C, " with accuracy score: ", svm_random_search.best_score_)
    if verbose: print()

    # MODEL TRAINING
    # --------------
    # this is not mandatory as normally RandomizedSearchCV could produce the "best" classifier while searching
    # for the best parameters.
    # model = svm_random_search.best_estimator_
    # However, I don't know how to retrieve the average and std values for the metrics
    # computed during CV for this particular best classifier.

    if verbose: print('training')
    svm_clf.set_params(C = best_C)
    scoring = {'acc': 'accuracy', 'prec': 'precision', 'rec': 'recall', 'f1':'f1'}

    start_time = time.time()

    #scores = cross_validate(svm_clf, X_train_b, y_train_b, scoring=scoring,
    #                     cv=nb_folds, return_train_score=True)

    scores = cross_validate(svm_clf, X_train_b, y_train_b, scoring=scoring,cv=nb_folds, return_train_score=True, fit_params={'sample_weight':coef_flattened})


    # scores_acc = cross_val_score(svm_clf, X_train_b, y_train_b, cv=nb_folds, scoring='accuracy')
    # scores_prec = cross_val_score(svm_clf, X_train_b, y_train_b, cv=nb_folds, scoring='precision')
    # scores_rec = cross_val_score(svm_clf, X_train_b, y_train_b, cv=nb_folds, scoring='recall')
    end_time=time.time()
    # svm_mean_accuracy_scores.append(np.mean(scores_acc))

    # print(scores.keys())
    # print("Mean accuracy values for ", nb_folds, " tests on training set: ", np.mean(scores['acc']))
    print("Average accuracy on training set : ", np.mean(scores['test_acc']), "+/-", np.std(scores['test_acc']))
    print("Average precision on training set : ", np.mean(scores['test_prec']), "+/-", np.std(scores['test_prec']))
    print("Average recall on training set : ", np.mean(scores['test_rec']), "+/-", np.std(scores['test_rec']))
    print("Average f1 on training set : ", np.mean(scores['test_f1']), "+/-", np.std(scores['test_f1']))
    print("Training time for ", nb_folds, "-fold cross-validation : ", (end_time - start_time), " seconds")
    if presentation: print("--------------------------------------------------")
    # MODEL TESTING
    # -------------
    if verbose: print('testing')
    #svm_clf.fit(X_train_b, y_train_b)
    pred = svm_random_search.best_estimator_.predict(X_test)
    score_prec = metrics.precision_score(y_test, pred)
    score_rec = metrics.recall_score(y_test, pred)
    score_acc = metrics.accuracy_score(y_test, pred)
    score_f1 = metrics.f1_score(y_test, pred)

    print("Accuracy on test set: ", score_acc)
    print("Precision on test set: ", score_prec)
    print("Recall on test set : ", score_rec)
    print("f1 on test set : ", score_f1)
    print("Weight vector : ", svm_random_search.best_estimator_.coef_)
    if presentation: print("--------------------------------------------------")



    # MODEL INTERPRETATION ON THE WHOLE DATASET
    # -----------------------------------------
    # train the svm classifier on all data to see the weight vector
    # and predict on the target dataset
    new_svm_clf = svm.LinearSVC(dual=False, penalty='l2', C=best_C)
    coef = kernel_mean_matching(dataSqlshare, X, kern='rbf', B=10)
#    coef = kernel_mean_matching(dataSqlshare, X)
    coef_flattened=coef.flatten()
    new_svm_clf.fit(X, y,sample_weight=coef_flattened)
    #new_svm_clf.fit(X, y)
    y_pred=new_svm_clf.predict(dataSqlshare)
    y_pred_conc=new_svm_clf.predict(X) #to measure agreement

    if verbose: print("SVM classifier weights : ")
    if verbose: print(np.array_str(new_svm_clf.coef_))
    print("SVM classifier weights on whole set: ")
    print(np.array_str(new_svm_clf.coef_))
    if presentation and verbose: print("--------------------------------------------------")
    sqlshareCustPath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/sqlShareCuts-'+str(sp).split('(')[0]+'.csv'
    with open(sqlshareCustPath, mode='w+') as cut_file:
        cut_writer = csv.writer(cut_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for y_p in y_pred:
            cut_writer.writerow([y_p])
    concatenateCustPath='/Users/marcel/Documents/RECHERCHE/STUDENTS/Willeme/concatenateCuts-transfer.csv'
    with open(concatenateCustPath, mode='w+') as cut_file_c:
        cut_writer = csv.writer(cut_file_c, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for y_p_c in y_pred_conc:
            cut_writer.writerow([y_p_c])
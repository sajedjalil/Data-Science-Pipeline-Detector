# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:40:37 CEST 2015

@author: Elena Cuoco
simple starting script, without the use of MNE
Thanks to @author: alexandrebarachant for his wornderful starting script


"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from glob import glob
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.metrics import roc_auc_score

from joblib import Parallel, delayed

#############function to read data###########

def prepare_data_train(fname):
    """ read and prepare training data """
    # Read data
    data = pd.read_csv(fname)
    # events file
    events_fname = fname.replace('_data','_events')
    # read event file
    labels= pd.read_csv(events_fname)
    clean=data.drop(['id' ], axis=1)#remove id
    labels=labels.drop(['id' ], axis=1)#remove id
    return  clean,labels

def prepare_data_test(fname):
    """ read and prepare test data """
    # Read data
    data = pd.read_csv(fname)
    return data

scaler= StandardScaler()
def data_preprocess_train(X):
    X_prep=scaler.fit_transform(X)
    #do here your preprocessing
    return X_prep
def data_preprocess_test(X):
    X_prep=scaler.transform(X)
    #do here your preprocessing
    return X_prep

def fit(X,y):
    # Do here you training
    clf = LogisticRegression()
    clf.fit(X,y)
    return clf

def predict(clf,X):
    # do here your prediction
    preds = clf.predict_proba(X)
    return np.atleast_2d(preds[:,clf.classes_==1])
    
# training subsample.if you want to downsample the training data
subsample = 100
#series used for CV
series = range(2,9)
#######columns name for labels#############
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

#######number of subjects###############
subjects = range(1,13)
auc_tot = []
pred_tot = []
y_tot = []
###loop on subjects and 8 series for train data + 2 series for test data
for subject in subjects:
    y_raw= []
    raw = []
    sequence = []
    ################ READ DATA ################################################
    
    for ser in series:
      fname =  '../input/train/subj%d_series%d_data.csv' % (subject,ser)
      data,labels=prepare_data_train(fname)
      raw.append(data)
      y_raw.append(labels)
      sequence.extend([ser]*len(data))

    X = pd.concat(raw)
    y = pd.concat(y_raw)
    #transform in numpy array
    #transform train data in numpy array
    X = np.asarray(X.astype(float))
    y = np.asarray(y.astype(float))
    sequence = np.asarray(sequence)


    ################ Train classifiers ########################################
    cv = LeaveOneLabelOut(sequence)
    pred = np.empty((X.shape[0],6))

    for train, test in cv:
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        #apply preprocessing
        X_train=data_preprocess_train(X_train)
        X_test=data_preprocess_test(X_test)
        clfs = Parallel(n_jobs=6)(delayed(fit)(X_train[::subsample,:],y_train[::subsample,i]) for i in range(6))
        preds = Parallel(n_jobs=6)(delayed(predict)(clfs[i],X_test) for i in range(6))
        pred[test,:] = np.concatenate(preds,axis=1)
    pred_tot.append(pred)
    y_tot.append(y)
    # get AUC
    auc = [roc_auc_score(y[:,i],pred[:,i]) for i in range(6)]     
    auc_tot.append(auc)
    print(auc)

pred_tot = np.concatenate(pred_tot)
y_tot = np.concatenate(y_tot)
global_auc = [roc_auc_score(y_tot[:,i],pred_tot[:,i]) for i in range(6)]

print('Global AUC : %.4f' % np.mean(global_auc))

auc_tot = np.asarray(auc_tot)
results = pd.DataFrame(data=auc_tot, columns=cols, index=subjects)
results.to_csv('results_cv_auc.csv')

plt.figure(figsize=(4,3))
results.mean(axis=1).plot(kind='bar')
plt.xlabel('Subject')
plt.ylabel('AUC')
plt.title('CV auc for each subject')
plt.savefig('cross_val_auc_subject.png' ,bbox_inches='tight')

plt.figure(figsize=(4,3))
results.mean(axis=0).plot(kind='bar')
plt.ylabel('AUC')
plt.title('CV auc for each class')
plt.savefig('cross_val_auc_class.png' ,bbox_inches='tight')

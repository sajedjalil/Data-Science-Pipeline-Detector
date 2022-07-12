import csv
from math import sqrt
import pandas as pd
import numpy as np

from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedShuffleSplit
#from sklearn import ensemble
from sklearn.ensemble import ExtraTreesClassifier

## LOADING THE DATA
print('Loading data...\n')
## training data
train = pd.read_csv('../input/train.csv')
target = train['target'].values
train = train.drop(['ID','target'],axis=1)

## test data (awaiting predictions)
test = pd.read_csv('../input/test.csv')
test_ID = test['ID'].values
test = test.drop(['ID'],axis=1)

## DATA PREPARATION
## Clearing the data sets
print('Clearing the data sets...\n')
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(), test.iteritems()):
    if train_series.dtype == 'object':
        ## for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        ## but now we have -1 values (NaN)
    elif train_series.dtype=='int':
        ## for int: fill in NaN
        ## train data set
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            train.loc[train_series.isnull(), train_name] = -9999 #fillna
        ## and test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -9999  #fillna
    else:
        tmp_len = len(train[train_series.isnull()])
        ## for float: fill in series.mean
        ## train data set
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = train_series.mean()
        #and test data set
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            #print "mean", test_series.mean()
            test.loc[test_series.isnull(), test_name] = test_series.mean()

print('The cleared DataFrames set are: \'X_train\', \'X_test\'\n')
X_train = train
X_test = test

## stratifiedShuffleSplit of the training set in a train and a validation part
print('Stratified Shuffle Split of the training set in a train and a validation part...')
sss = StratifiedShuffleSplit(target, n_iter=1, test_size=0.5, random_state=1)
for train_idx, valdt_idx in sss:    
    X_train, X_valdt = X_train.loc[train_idx,], X_train.loc[valdt_idx,]
    y_train, y_valdt = target[train_idx], target[valdt_idx]
print('The train data set is: (X_train, y_train)')
print('The validation data set is: (X_valdt, y_valdt)')

## TRAIN the ExtraTreesClassifier (OPTIMAL PARAMETERS?)
print('Creating an ExtraTreesClassifier object...')
n_estimators = 700
max_features= 66
min_samples_split= 2
print('with... \'n_estimators\' = %.d' %n_estimators)
print('with... \'max_features\' = %.d' %max_features)
print('with... \'min_samples_split\' = %.d' %min_samples_split)
print('with... \'min_samples_leaf\' = %.d\n' %min_samples_split)

extrTrClassfr = ExtraTreesClassifier(n_estimators=n_estimators,
                                     max_features= max_features,
                                     criterion= 'entropy',
                                     min_samples_split= min_samples_split,
                                     max_depth= max_features,
                                     min_samples_leaf= min_samples_split, ## the default is 1 (note we use more than one sample for the split)
                                     class_weight='balanced_subsample',
                                     random_state=1,
                                     verbose=1)
    
print('Training the ExtraTreesClassifier...\n')
extrTrClassfr.fit(X_train, y_train)

## ESTIMATE MODEL'S PREDICTIVE PERFORMANCE USING THE VALIDATION SET
print('Making predictions for the validation set...\n')
valdt_PredProb = extrTrClassfr.predict_proba(X_valdt)
    
print('Estimating the predictive performance:')
log_loss = log_loss(y_valdt, valdt_PredProb)
print('\'log_loss\'[validation set]: %.5f' %log_loss)
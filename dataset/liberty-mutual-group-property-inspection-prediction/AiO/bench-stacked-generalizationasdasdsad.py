# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 23:50:55 2015

@author: Justfor <justformaus@gmx.de>

Purpose: This script tries to implement a technique called stacking/stacked generalization.
I made this a runnable script available because I found that there isn't really any
readable code that demonstrates this technique. 
======================================================================================================
Summary:

Just to test an implementation of stacking/stacked generalization.
Using a cross-validated ExtraTrees Regressor, Random forest and Gradient Boosting Regressor.
Leaderboard: 0.361412
Improvements should be very easy.

This code is heavily inspired from the classification code shared by Emanuele (https://github.com/emanuele)
and by Eric Chio "log0" <im.ckieric@gmail.com>,  but I have made it for regression, cleaned it up to
 make it available for easy download and execution.

======================================================================================================
Methodology:

Three classifiers (ExtraTreesRegressor, RandomForestRegressor and a GradientBoostingRegressor)
are built to be stacked by a RidgeCVRegressor in the end.

Some terminologies first, since everyone has their own, I'll define mine to be clear:
- DEV SET, this is to be split into the training and validation data. It will be cross-validated.
- TEST SET, this is the unseen data to validate the generalization error of our final classifier. This
set will never be used to train.
When DEVELOPMENT is set True, then cross validation and evaluation takes place.
Otherwise (DEVELOPMENT=False) a submission file is generated.

======================================================================================================
Data Set Information:
Kaggle Competition LMP Property Inspection
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from datetime import datetime
                                 
# Source of good version: https://www.kaggle.com/c/ClaimPredictionChallenge/forums/t/703/code-to-calculate-normalizedgini    
def gini(actual, pred, cmpcol = 0, sortcol = 1):
     assert( len(actual) == len(pred) )
     all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
     all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
     totalLosses = all[:,0].sum()
     giniSum = all[:,0].cumsum().sum() / totalLosses
     giniSum -= (len(actual) + 1) / 2.
     return giniSum / len(actual)
 
def normalized_gini(a, p):
     return gini(a, p) / gini(a, a)


def prepare_data():
    # load train data
    train    = pd.read_csv('../input/train.csv')
    test     = pd.read_csv('../input/test.csv')
    
    labels   = train.Hazard   
    test_ind = test.ix[:,'Id']
    train.drop('Hazard', axis=1, inplace=True)
    train_ind = train.ix[:,'Id']
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)
    Dropcols = ['T2_V10', 'T2_V7','T1_V13', 'T1_V10']
    train.drop( Dropcols, axis = 1, inplace=True )
    test.drop( Dropcols, axis = 1, inplace=True )
    
    catCols=['T1_V4', 'T1_V5','T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12', 'T1_V15',
     'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5', 'T2_V11', 'T2_V12', 'T2_V13']
  
    trainNumX=train.drop(catCols, axis=1)
    trainCatVecX = pd.get_dummies(train[catCols])   
    trainX = np.hstack((trainCatVecX,trainNumX))

    testNumX=test.drop(catCols, axis=1)
    testCatVecX = pd.get_dummies(test[catCols])  
    testX = np.hstack((testCatVecX,testNumX))
 
    return trainX, labels, train_ind, testX, test_ind
     
if __name__ == '__main__':    
    DEVELOP = False

    SEED = 42   
    np.random.seed(SEED)

    X, Y, idx, testX, testidx = prepare_data()

 
    submission = pd.DataFrame(testX)
    submission.to_csv("testwq.csv") 

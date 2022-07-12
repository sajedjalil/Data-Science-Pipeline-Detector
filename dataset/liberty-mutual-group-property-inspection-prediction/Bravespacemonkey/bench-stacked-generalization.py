# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from datetime import datetime


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
    print(idx)
    # for i in range(len(X[1])):
    #     print(X[1][i])
    
    # if (DEVELOP==True):
    #     # The DEV SET will be used for all training and validation purposes
    #     # The TEST SET will never be used for training, it is the unseen set.
    #     dev_cutoff = int(round(len(Y) * 4/5))
    #     X_dev = X[:dev_cutoff]
    #     Y_dev = Y[:dev_cutoff]
    #     X_test = X[dev_cutoff:]
    #     Y_test = Y[dev_cutoff:]              
    # else:    # else submit    
    #     X_dev = X
    #     Y_dev = Y
    #     X_test = testX
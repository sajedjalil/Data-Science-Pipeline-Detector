# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import scipy
from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-


#===================================prep data==================================


def load_data(train_file, test_file):
    '''
    Takes as input:
        train_file: the training csv data filename
        train_file: the test csv data filename
    What it does:
        makes use of the pandas read_csv method to convert the csv data to
        pandas dataframes
    Returns:
        training data DataFrame
        test data DataFrame
        '''
    return pd.read_csv(train_file), pd.read_csv(test_file)

def zap_empties(train, test):
    '''
    Takes as input:
        train: the training factor data in the form a pandas DataFrame
        test: the test factor data in the form a pandas DataFrame
    What it does:
        finds all columns in the training factor data where the standard
        deviation is zero. This implies those columns are constant in the
        training factor data and therefore cannot be used for learning. The
        columns are removed from both the training and test factor data
    Returns:
        training data DataFrame
        test data DataFrame
        '''
    sd = train.std()
    empties = sd[sd==0.0].index
    return train.drop(empties,1), test.drop(empties,1)

def zap_dependencies(train, test, verbose=False):
    '''
    Takes as input:
        train: the training factor data in the form a pandas DataFrame
        test: the test factor data in the form a pandas DataFrame
        verbose: if TRUE then prints out dependent column names
    What it does:
        Converts the training factor data to a numpy matrix then performs a QR
        decomposition on it. Dependent columns are identified as all those that
        do not have pivots (i.e., are within a certain tolerence of zero where
        the pivot should be). These columns are then removed from the training
        and test factor data
    Returns:
        training data DataFrame
        test data DataFrame
    '''
    dependencies = []
    feature_cols = train.columns
    Q, R = np.linalg.qr(np.matrix(train))
    indep_locs = np.where(abs(R.diagonal())>1e-7)[1]
    for i, col in enumerate(feature_cols):
        if i not in indep_locs:
            dependencies.append(col)
            if verbose:
                print (col)
    return train.drop(dependencies,1), test.drop(dependencies,1)

def acquire_data(train_file, test_file, target_col, id_col, verbose=False):
    '''
    Takes as input:
        train: the training data in the form a pandas DataFrame
        test: the test data in the form a pandas DataFrame
        target_col: the name of the target (output) column
        id_col: the name of the ID column
        verbose: fed to zap_dependencies function
    What it does:
        Calls upon helper functions to read in and manipulate data. Splits
        the train and test data into X and y groupings
    Returns:
        training data features
        training data labels
        test data features
        test data IDs
    '''
    df_train, df_test = load_data(train_file, test_file)
    feature_cols = df_train.columns.difference([target_col, id_col])
    y_train = df_train[target_col]
    id_test = df_test[id_col]
    X_train, X_test = df_train[feature_cols], df_test[feature_cols]
    if verbose:
        print (X_train.shape, X_test.shape)
    X_train, X_test = zap_empties(X_train, X_test)
    if verbose:
        print (X_train.shape, X_test.shape)
    X_train, X_test = zap_dependencies(X_train, X_test, verbose)
    if verbose:
        print (X_train.shape, X_test.shape)
    return X_train, y_train, X_test, id_test

#==============================================================================

target_col = 'TARGET'
id_col = 'ID'

X_train, y_train, X_test, id_test = acquire_data('../input/train.csv', '../input/test.csv',
                                    target_col, id_col, verbose=False)

#==============================================================================


booster = XGBClassifier(
                        n_estimators        =   336,
                        learning_rate       =   0.0235,
                        max_depth           =   4,
                        subsample           =   0.680,
                        colsample_bytree    =   0.996
                        )


#0.839930776843



X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train,
                                    test_size=0.25, stratify=y_train)

booster.fit(X_fit, y_fit, eval_metric="auc", eval_set=[(X_val, y_val)])

# predicting
y_pred = booster.predict_proba(X_test)[:,1]

submission = pd.DataFrame({'TARGET':y_pred}, index=id_test)
submission.to_csv('submission.csv')

print('Completed!')

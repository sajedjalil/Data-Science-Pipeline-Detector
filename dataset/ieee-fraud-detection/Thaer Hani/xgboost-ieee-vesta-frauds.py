# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np # linear algebra
np.set_printoptions(precision=2)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame
#Parallelism
import joblib
from joblib import Parallel, delayed,parallel_backend
import datetime

# Preprocessing, modelling and evaluating
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error, mean_absolute_error
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Preprocessing, modelling and evaluating
from sklearn.preprocessing import Imputer,MaxAbsScaler,StandardScaler
## Hyperopt modules
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from functools import partial

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df_trans = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
df_test_trans = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

df_id = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
df_test_id = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')

df_train = df_trans.merge(df_id, how='left', left_index=True, right_index=True)
df_test = df_test_trans.merge(df_test_id, how='left', left_index=True, right_index=True)


## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

many_null_cols = [col for col in df_train.columns if df_train[col].isnull().sum() / df_train.shape[0] > 0.9]
many_null_cols_test = [col for col in df_test.columns if df_test[col].isnull().sum() / df_test.shape[0] > 0.9]

cols_to_drop = list(set(many_null_cols + many_null_cols_test))
len(cols_to_drop)

df_train = df_train.drop(cols_to_drop, axis=1)
df_test = df_test.drop(cols_to_drop, axis=1)

df_train = df_train.fillna(-999)
df_test = df_test.fillna(-999) 

for f in df_train.columns:
    if df_train[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_train[f].values) + list(df_test[f].values))
        df_train[f] = lbl.transform(list(df_train[f].values))
        df_test[f] = lbl.transform(list(df_test[f].values))
        

threshold = 0.98
    
# Absolute value correlation matrix
corr_matrix = df_train[df_train['isFraud'].notnull()].corr().abs()

# Getting the upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove.' % (len(to_drop)))
df_train = df_train.drop(columns = to_drop)
df_test = df_test.drop(columns = to_drop)


X_train = df_train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT'], axis=1)
y_train = df_train.sort_values('TransactionDT')['isFraud']
X_test = df_test.sort_values('TransactionDT').drop(['TransactionDT'], axis=1)

df_test = df_test[["TransactionDT"]]

features = list(df_train.columns[1:])  #la colonne 0 est le quote_conversionflag  
'''
parameters = {
              'objective':['binary:logistic'],
              'learning_rate': [0.05],
              'max_depth': [9],
              'silent': [1],
              'subsample': [0.9],
              'colsample_bytree': [1.0],
              'n_estimators': [500], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'random_state': [2019],
              'reg_alpha':  [0.01],
              'reg_lambda': [0.3],
              'gamma': [0.1],
              'num_leaves': [20],       
              'min_child_samples':  [10, 80, 3],
              'feature_fraction': [ 0.6],
              'bagging_fraction': [ 0.7]
            }
'''
parameters = {
                'max_depth': [9]

            }
X_train = reduce_mem_usage(X_train)
X_test = reduce_mem_usage(X_test)



tscv = TimeSeriesSplit(n_splits=2)
#==============================================XGBClassifier=============================
model = xgb.XGBClassifier()
clf = GridSearchCV(model, parameters, 
                cv=tscv, 
                scoring='roc_auc',
                verbose=2, refit=True)
print('********************** XGBoost GridSearch CV abou to starts ***************************')
print(datetime.datetime.now())
print()
print('------------ fit method starts ---------------')

#Parallel(n_jobs=7)
#(delayed(
clf.fit(X_train, y_train)
#))
print()
print('------------ fit method starts ---------------')
print()
print('********************** XGBoost GridSearch CV ends ***************************')
print(datetime.datetime.now())
print()

print("Best score: %0.3f" % clf.best_score_)
print("Best parameters set:")
best_parameters=clf.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

test_probs = clf.predict_proba(X_test)[:,1]

sample = pd.read_csv('../input/sample_submission.csv')
sample.isFraud = test_probs
sample.to_csv("xgboost_best_parameter_submission.csv", index=False)
#=============================KNeighborsClassifier==============================================
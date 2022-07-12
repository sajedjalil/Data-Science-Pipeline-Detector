# Light Gradient Boosting ML taken from https://www.kaggle.com/nakshatrasingh/jane-street-with-boosting-algorithms

import os
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cudf
import cupy as cp
import janestreet
import xgboost as xgb
import math
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, precision_score,recall_score
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix

train_cudf = cudf.read_csv('/kaggle/input/jane-street-market-prediction/train.csv') 
train = train_cudf.to_pandas()
del train_cudf

train = train[train['weight'] > 0]
train.reset_index(inplace=True, drop = True)

train['action'] = (train['resp'] > 0).astype('int') 

features = [c for c in train.columns if 'feature' in c]
X_train = train.loc[:, features]
y_train = train.loc[:, 'action']

X_reward = train.loc[:,'weight'] * train.loc[:,'resp']
X_day = train.loc[:,'date']

del train

# CV with TimeSeriesSplit

params = {
    'objective' : "binary",
    'n_estimators': 100,
    'max_depth': 10,
    'learning_rate': 0.1,
    'subsample' : 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 0.01,
    'is_unbalance': 'true',
    'missing': None,
    'random_state': 42,
    'device': 'gpu',
    'verbosity': 1    
 }

clf = LGBMClassifier()
clf.set_params(**params)

tscv = TimeSeriesSplit(n_splits=2)

def utility_gain(pred_val,test_index, X_reward,X_day):

    uniqueValues = X_day.loc[test_index,].unique()
    min_day = uniqueValues.min().astype('int32')
    max_day = uniqueValues.max().astype('int32')
    nro_days = max_day - min_day +1
    
    pi = np.zeros(nro_days)
    pi2 = np.zeros(nro_days)
    
    for j in range(min_day, max_day):        
        index = X_day.index[X_day == j].intersection(test_index)
        m = X_reward[index] * pred_val[index+min_day]
        pi[j-min_day] = np.sum(m) 
        pi2[j-min_day] = pi[j - min_day] ** 2

    t = (np.sum(pi)/ math.sqrt(np.sum(pi2))) * math.sqrt(250/nro_days)
    utility = min(max(t,0),6) * np.sum(pi)

    return utility

for train_index, test_index in tscv.split(X_train, y_train):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_tr, X_te = X_train.loc[train_index], X_train.loc[test_index]
    y_tr, y_te = y_train.loc[train_index], y_train.loc[test_index]

    clf.fit(X_tr, y_tr)
    pred_val = clf.predict(X_te).round().astype(int)
    pred_val = pd.Series(pred_val)
    pred_val.index = X_te.index
    utility = utility_gain(pred_val,test_index,X_reward,X_day)
    print("%.2f" % utility)

    accuracy = accuracy_score(y_te, pred_val)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    precision = precision_score(y_te, pred_val)
    print("Precision: %.2f%%" % (precision * 100.0)) 
    
    recall = recall_score(y_te, pred_val)
    print("Recall: %.2f%%" % (recall*100.00))
   
env = janestreet.make_env()
iter_test = env.iter_test() # an iterator which loops over the test set

for (test_df, sample_prediction_df) in iter_test:
    X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
    y_preds = clf.predict(X_test)
    sample_prediction_df.action = y_preds
    env.predict(sample_prediction_df)


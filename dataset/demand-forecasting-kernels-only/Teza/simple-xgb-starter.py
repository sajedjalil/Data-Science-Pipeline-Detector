# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')

train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.dayofweek
train['year'] = train['date'].dt.year

test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.dayofweek
test['year'] = test['date'].dt.year

col = [i for i in test.columns if i not in ['date','id']]
y = 'sales'

train_x, train_cv, y, y_cv = train_test_split(train[col],train[y], test_size=0.2, random_state=2018)

def XGB_regressor(train_X, train_y, test_X, test_y, feature_names=None, seed_val=2017, num_rounds=500):
    param = {}
    param['objective'] = 'reg:linear'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['eval_metric'] = 'mae'
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)
        
    return model    
    
    
model = XGB_regressor(train_X = train_x, train_y = y, test_X = train_cv, test_y = y_cv)
y_test = model.predict(xgb.DMatrix(test[col]), ntree_limit = model.best_ntree_limit)

sample['sales'] = y_test
sample.to_csv('simple_xgb_starter.csv', index=False)


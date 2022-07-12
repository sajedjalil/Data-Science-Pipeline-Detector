# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import gc
import time
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import csv

dtypes = {'ip': 'uint32', 'app': 'uint16', 'device': 'uint16', 'os': 'uint16', 'channel': 'uint16', 'is_attributed' : 'uint8',}
train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']

train_df = pd.read_csv("../input/train.csv", skiprows=range(1,144903891), nrows=65000000, dtype=dtypes, usecols=train_columns)
test_df = pd.read_csv("../input/train.csv", nrows=19000000, dtype=dtypes, usecols=train_columns)

y_col = 'is_attributed'
Y = train_df[y_col].values
del train_df[y_col]
X = train_df.values

Y_valid = test_df[y_col].values
del test_df[y_col]
X_valid = test_df.values

X = np.delete(X, 5, 1)
X_valid = np.delete(X_valid, 5, 1)
train_lgb = lgb.Dataset(X, label=Y)
test_lgb = lgb.Dataset(X_valid, Y_valid, reference=train_lgb)

lgb_params = {'boosting_type': 'gbdt','objective': 'binary','subsample_for_bin': 200000, 'min_split_gain': 0,'reg_alpha': 0,'reg_lambda': 0, 'nthread': 4,'verbose': 0,'metric':'auc','learning_rate': 0.15,'num_leaves': 7,  'max_depth': 3, 'min_child_samples': 100, 'max_bin': 100,'subsample': 0.7, 'subsample_freq': 1, 'colsample_bytree': 0.9,'scale_pos_weight':99}
evals_results = {}
num_boost_round = 200
early_stopping_rounds = 30

booster = lgb.train(lgb_params, train_lgb, valid_sets=[test_lgb], evals_result=evals_results, num_boost_round=num_boost_round,early_stopping_rounds=early_stopping_rounds,verbose_eval=1)
test_predict = pd.read_csv('../input/test.csv', dtype=dtypes)

test_predict = np.delete(test_predict.values, 6, 1)
test_id = test_predict[:, 0]
test = np.delete(test_predict, 0, 1)
predictions = booster.predict(test)
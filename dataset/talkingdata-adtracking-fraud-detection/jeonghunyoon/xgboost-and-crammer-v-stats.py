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

# This kernel have improvement from https://www.kaggle.com/alexanderkireev/deep-learning-support-9663

import pandas as pd
import numpy as np
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['OMP_NUM_THREADS'] = '8'

# Features
IP = 'ip'
APP = 'app'
DEVICE = 'device'
OS = 'os'
CHANNEL = 'channel'
CLICK_TIME = 'click_time'
ATTRIBUTED_TIME = 'attributed_time'
IS_ATTRIBUTED = 'is_attributed'
CLICK_ID='click_id'

# New features related to time
DAY_OF_WEEK = 'day_of_week'
DAY_OF_YEAR = 'day_of_year'

def timeFeatures(df):
    # Make some new features with click_time column
    df[DAY_OF_WEEK] = pd.to_datetime(df[CLICK_TIME]).dt.dayofweek
    df[DAY_OF_YEAR] = pd.to_datetime(df[CLICK_TIME]).dt.dayofyear
    df.drop([CLICK_TIME], axis=1, inplace=True)
    return df
    
TRAIN_COLUMNS = [IP, APP, DEVICE, OS, CHANNEL, CLICK_TIME, ATTRIBUTED_TIME, IS_ATTRIBUTED]
TEST_COLUMNS = [IP, APP, DEVICE, OS, CHANNEL, CLICK_TIME, CLICK_ID]

dtypes = {
    IP : 'int32',
    APP : 'int16',
    DEVICE : 'int16',
    OS : 'int16',
    CHANNEL : 'int16',
    IS_ATTRIBUTED : 'int8',
    CLICK_ID : 'int32'
}

# Train set
train_set = pd.read_csv('../input//train.csv', 
                        skiprows = range(1, 123903891), 
                        nrows=61000000, 
                        usecols=TRAIN_COLUMNS, 
                        dtype=dtypes)
# Test set
test_set = pd.read_csv('../input/test.csv', 
                       usecols=TEST_COLUMNS, 
                       dtype=dtypes)
                       
# Checkin Cremer V stats
# Method for cheking Cramer V stat
import scipy.stats as ss

def get_cramers_stat(confusion_matrix):
    confusion_matrix = np.array(confusion_matrix)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    # min(confusion_matrix.shape)-1 : 자유도
    cramers_stat = np.sqrt(phi2 / (min(confusion_matrix.shape)-1))
    return cramers_stat
    
# Ip Crammer V stat
print(get_cramers_stat(pd.crosstab(train_set.ip, train_set.is_attributed)))
# App Crammer V stat
print(get_cramers_stat(pd.crosstab(train_set.app, train_set.is_attributed)))
# Device Crammer V stat
print(get_cramers_stat(pd.crosstab(train_set.device, train_set.is_attributed)))
# OS Crammer V stat
print(get_cramers_stat(pd.crosstab(train_set.os, train_set.is_attributed)))
# Channel Crammer V stat
print(get_cramers_stat(pd.crosstab(train_set.channel, train_set.is_attributed)))

# Split Y 
y = train_set[IS_ATTRIBUTED]
train_set.drop([IS_ATTRIBUTED], axis=1, inplace=True)

# Sub dataframe is for submission.
sub = pd.DataFrame()
sub[CLICK_ID] = test_set[CLICK_ID]

nrow_train = train_set.shape[0]

merge = pd.concat([train_set, test_set])

del train_set, test_set
gc.collect()

# New features using group by
CLICKS_BY_IP = 'clicks_by_ip'
CLICKS_BY_IP_APP = 'clicks_by_ip_app'

# Count the number of clicked channels by ip
ip_count = merge.groupby([IP])[CHANNEL].count().reset_index()\
    .rename(columns = {CHANNEL : CLICKS_BY_IP})
merge = pd.merge(merge, ip_count, on=IP, how='left', sort=False)
merge[CLICKS_BY_IP] = merge[CLICKS_BY_IP].astype('int16')

# IP가 특정 app을 얼마나 많은 장소에서 클릭했는가?
ip_app_count = merge.groupby(by=[IP, APP])[CHANNEL].count().reset_index()\
    .rename(columns=({CHANNEL: CLICKS_BY_IP_APP}))
merge = pd.merge(merge, ip_app_count, on=[IP, APP], how='left', sort=False)
merge[CLICKS_BY_IP_APP] = merge[CLICKS_BY_IP_APP].astype('int16')

# Drop columns not necessary
# IP?
merge.drop([ATTRIBUTED_TIME, CLICK_ID], axis=1, inplace=True)

# Adding new features
merge = timeFeatures(merge)

train_set = merge[:nrow_train]
test_set = merge[nrow_train:]

# Train set, y를 join해서, clicks_by_ip와의 correlation을 구해본다.
correlation_df = pd.concat([train_set[[CLICKS_BY_IP, CLICKS_BY_IP_APP]], y], axis=1)
correlation = correlation_df.corr()
sns.heatmap(correlation, cmap='viridis', annot=True, linewidth=3)

del correlation_df, correlation
gc.collect()

# Xgboost model
params = {
    'eta': 0.3,
    'tree_method': 'hist',
    'grow_policy': 'lossguide',
    'max_leaves': 1400,
    'max_depth': 0,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    'min_child_weight': 0,
    'alpha': 4,
    'objective': 'binary:logistic',
    'scale_pos_weight': 9,
    'eval_metric': 'auc',
    'nthread': 8,
    'random_state': 99,
    'silent': True
}

from sklearn.model_selection import train_test_split

import xgboost as xgb
from xgboost import plot_importance

is_valid = False

if(is_valid == True):
    x_train, x_valid, y_train, y_valid = train_test_split(train_set, y, test_size=0.1, random_state=99)
    dtrain = xgb.DMatrix(x_train, y_train)
    dvalid = xgb.DMatrix(x_valid, y_valid)
    del x_train, x_valid, y_train, y_valid
    gc.collect()
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    model = xgb.train(params, dtrain, 200, watchlist, maximize=True, early_stopping_rounds=25, verbose_eval=5)
    del dvalid
else:
    dtrain = xgb.DMatrix(train_set, y)
    del train_set, y
    gc.collect()
    watchlist = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, 30, watchlist, maximize=True, verbose_eval=1)
    
del dtrain
gc.collect()

plot_importance(model)

dtest = xgb.DMatrix(test_set)
sub[IS_ATTRIBUTED] = model.predict(dtest, ntree_limit=model.best_ntree_limit)

del dtest
gc.collect()

sub.to_csv('submission.csv', index=False)
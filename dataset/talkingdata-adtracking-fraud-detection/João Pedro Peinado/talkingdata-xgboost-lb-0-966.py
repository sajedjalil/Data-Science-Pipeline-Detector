# This is an improvement based on kernel from Pranav Pandya
# the kernel is available in this link: https://www.kaggle.com/pranav84/xgboost-on-hist-mode-ip-addresses-dropped

import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb

# Change this for validation with 10% from train
using_test = True

path = '../input/'

def dataPreProcessTime(df):
    # Make some new features with click_time column
    df["datetime"] = pd.to_datetime(df['click_time'])
    df['dow']      = df['datetime'].dt.dayofweek
    df['woy']      = df['datetime'].dt.week
    df['moy']      = df['datetime'].dt.month
    df.drop(['click_time, datetime'], axis=1, inplace=True)
    return df

start_time = time.time()

columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }

# Read the last lines because they are more impacting in training than the starting lines
train = pd.read_csv(path+"train.csv", skiprows=range(1,149903891), nrows=35000000, usecols=columns, dtype=dtypes)
test = pd.read_csv(path+"test.csv")

print('[{}] Finished to load data'.format(time.time() - start_time))

train = dataPreProcessTime(train)
test = dataPreProcessTime(test)

# Drop the IP and the columns from target
y = train['is_attributed']
train.drop(['is_attributed'], axis=1, inplace=True)

# Drop IP and ID from test rows
sub = pd.DataFrame()
sub['click_id'] = test['click_id']
test.drop(['click_id'], axis=1, inplace=True)

# Some feature engineering
nrow_train = train.shape[0]
merge = pd.concat([train, test])

# Count the number of clicks by ip
ip_count = merge.groupby('ip')['app'].count().reset_index()
ip_count.columns = ['ip', 'clicks_by_ip']
merge = pd.merge(merge, ip_count, on='ip', how='left', sort=False)
merge.drop('ip', axis=1, inplace=True)

train = merge[:nrow_train]
test = merge[nrow_train:]

print('[{}] Start XGBoost Training'.format(time.time() - start_time))

# Set the params(this params from Pranav kernel) for xgboost model
params = {'eta': 0.6,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,  
          'max_depth': 0, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'binary:logistic', 
          'scale_pos_weight':9,
          'eval_metric': 'auc', 
          'nthread':8,
          'random_state': 99, 
          'silent': True}

if (using_test == False):
    # Get 10% of train dataset to use as validation
    x1, x2, y1, y2 = train_test_split(train, y, test_size=0.1, random_state=99)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 300, watchlist, maximize=True, early_stopping_rounds = 50, verbose_eval=10)
else:
    watchlist = [(xgb.DMatrix(train, y), 'train')]
    model = xgb.train(params, xgb.DMatrix(train, y), 15, watchlist, maximize=True, verbose_eval=1)

print('[{}] Finish XGBoost Training'.format(time.time() - start_time))

sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
sub.to_csv('xgb_sub.csv',index=False)
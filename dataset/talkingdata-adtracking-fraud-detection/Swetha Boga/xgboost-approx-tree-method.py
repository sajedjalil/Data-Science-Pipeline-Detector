# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import gc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

path = '../input/'
traincolumns = ['ip','app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
train = pd.read_csv(path+'train.csv',skiprows=range(1,149903891), nrows=10000000, usecols = traincolumns)
test = pd.read_csv(path+'test.csv')
sub = pd.DataFrame()
sub['click_id'] = test['click_id']
train = train.dropna()

def preprocessClicktime(df):
	# Make some new features with click_time column
	df["datetime"] = pd.to_datetime(df['click_time'])
	df['dow']      = df['datetime'].dt.dayofweek
	df['woy']      = df['datetime'].dt.week
	df['day'] 	   = df['datetime'].dt.day
	df['hour'] 	   = df['datetime'].dt.hour
	df['minute']   = df['datetime'].dt.minute
	df['second']   = df['datetime'].dt.second
	return df

train = preprocessClicktime(train)
train = train.drop(['click_time','datetime'],axis=1)
test = preprocessClicktime(test)
test = test.drop(['click_id','click_time','datetime'],axis=1)

y = train['is_attributed']
train = train.drop(['is_attributed'], axis=1)

# Some feature engineering
nrow_train = train.shape[0]
merge = pd.concat([train, test])
del train, test
gc.collect()

# Count the number of clicks by ip
ip_count = merge.groupby('ip')['app'].count().reset_index()
ip_count.columns = ['ip', 'clicks_by_ip']
merge = pd.merge(merge, ip_count, on='ip', how='left', sort=False)
merge.drop('ip', axis=1, inplace=True)

train = merge[:nrow_train]
test = merge[nrow_train:]
del merge
gc.collect()

params = {'eta': 0.3, # learning rate
          'tree_method': "auto", 
          'max_depth': 4, 
          'subsample': 0.8, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'binary:logistic', 
          'scale_pos_weight':9,
          'eval_metric': 'auc', 
          'random_state': 99, 
          'silent': True}
          
x1, x2, y1, y2 = train_test_split(train, y, test_size=0.1, random_state=99)
watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 300, watchlist, maximize=True, early_stopping_rounds = 50, verbose_eval=10)

sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
sub.to_csv('xgb_sub6.csv',index=False) # LB score 0.9633

# Any results you write to the current directory are saved as output.
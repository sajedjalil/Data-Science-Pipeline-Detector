"""
XGBoost on Hist mode with lossguide to give lightgbm characteristic (reduces runtime)

Improvements based on following excellent kernels:
- João Pedro Peinado's XGB starter script: https://www.kaggle.com/joaopmpeinado/xgboost-lb-0-951
- Andy Harless' script for memory optimization: https://www.kaggle.com/aharless/jo-o-s-xgboost-with-memory-usage-enhancements
- Ravi Teja's fe script: https://www.kaggle.com/rteja1113/lightgbm-with-count-features?scriptVersionId=2815638
"""


import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from xgboost import plot_tree
from xgboost import plot_importance
import matplotlib.pyplot as plt
import gc

path = '../input/'

dtypes = {'ip'            : 'uint32',
          'app'           : 'uint16',
          'device'        : 'uint16',
          'os'            : 'uint16',
          'channel'       : 'uint16',
          'is_attributed' : 'uint8',
          'click_id'      : 'uint32'
          }

print('loading train data...')

train_df = pd.read_csv(path+"train.csv", skiprows=range(1,147403891),  nrows=37500000, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
# total observations: 184,903,891

y = train_df['is_attributed']
train_df.drop(['is_attributed'], axis=1, inplace=True)

print('loading test data...')
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id']
test_df.drop(['click_id'], axis=1, inplace=True)

len_train = len(train_df)
train_df=train_df.append(test_df)

del test_df
gc.collect()

print('Extracting day and hour...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day']  = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df['dow']  = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8')

gc.collect()

# # of clicks for each ip-day-hour combination
print('grouping by ip-day-hour combination...')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
del gp
gc.collect()

# # of clicks for each ip-app combination
print('group by ip-app combination...')
gp = train_df[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp, on=['ip','app'], how='left')
del gp
gc.collect()

# # of clicks for each ip-app-os combination
print('group by ip-app-os combination...')
gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()


print("vars and data type: ")
train_df.drop(['ip'], axis=1, inplace=True)
train_df.drop(['click_time'], axis=1, inplace=True)
train_df.info()

train_df['qty'] = train_df['qty'].astype('uint16')
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

#split back
test  = train_df[len_train:]
train = train_df[:(len_train)]

gc.collect()

print("train size: ", len(train))
print("test size : ", len(test))

start_time = time.time()

"""
XGBoost parameters tuning guide:
https://github.com/dmlc/xgboost/blob/master/doc/how_to/param_tuning.md
https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
"""
params = {'eta': 0.6, 
          'tree_method': "hist",      # Fast histogram optimized approximate greedy algorithm. 
          'grow_policy': "lossguide", # split at nodes with highest loss change
          'max_leaves': 1400,         # Maximum number of nodes to be added. (for lossguide grow policy)
          'max_depth': 0,             # 0 means no limit (useful only for depth wise grow policy)
          'subsample': 0.9,           
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,       # The larger, the more conservative the algorithm will be
          'alpha':4,                  # L1 regularization on weights | default=0 | large value == more conservative model
          'objective': 'binary:logistic', 
          'scale_pos_weight':9,       # because training data is extremely unbalanced 
          'eval_metric': 'auc', 
          'nthread':8,
          'random_state': 84, 
          'silent': True}

x1, x2, y1, y2 = train_test_split(train, y, test_size=0.05, random_state=84)

del train
gc.collect()

# watch list to observe the change in error in training and holdout data
watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]

model = xgb.train(params, xgb.DMatrix(x1, y1), 50, watchlist, maximize=True, early_stopping_rounds = 5, verbose_eval=1)

del x1, x2, y1, y2
gc.collect()

print('[{}]: Training time for Histogram Optimized XGBoost model'.format(time.time() - start_time))


print("predicting...")
sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
sub.to_csv('sub_xgb_hist_pos_weight.csv',index=False)


#Model evaluation
print("Extract feature importance matrix")
plot_importance(model)
plt.gcf().savefig('xgb_fe.png')

print("finished...")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import time
import gc
import os

train = pd.read_csv('../input/train_V2.csv')
subm = pd.read_csv('../input/sample_submission_V2.csv')

train.dtypes

target = train['winPlacePerc']
train.drop(['Id','winPlacePerc','matchType','rankPoints'], axis=1, inplace=True)

def featureEngineering(df):
    df_size = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    df_mean = df.groupby(['matchId','groupId']).mean().reset_index()
    df_max = df.groupby(['matchId','groupId']).max().reset_index()
    df_min = df.groupby(['matchId','groupId']).min().reset_index()
    df_match_mean = df.groupby(['matchId']).mean().reset_index()
    df_train_max_PG = df.groupby(['matchId','groupId'])['kills'].count().reset_index().groupby('matchId')['kills'].max().reset_index()
    df_train_max_PG.columns = ['matchId','max_players_in_group']
    
    df = pd.merge(df, df_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
    df = pd.merge(df, df_max, suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
    df = pd.merge(df, df_min, suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
    df = pd.merge(df, df_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    df = pd.merge(df, df_size, how='left', on=['matchId', 'groupId'])
    df = pd.merge(df, df_train_max_PG, how='left', on=['matchId'])
    return df

# print('[{}] Start making some feature engineering...'.format(time.time() - start_time))

train = featureEngineering(train)

train.drop(['matchId','groupId'], axis=1, inplace=True)

train_columns = list(train.columns)
train_columns_new = []
for name in train_columns:
    if '_' in name:
        train_columns_new.append(name)  
#print(train_columns_new)

train = train[train_columns_new]

train.dtypes.value_counts()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train, target, test_size = 0.2, random_state = 2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

del train

gc.collect()

y_test.fillna(0, inplace = True)


params = {
    'boosting_type':'gbdt',
    'learning_rate': 0.1, 
    'max_depth': -1,
    'num_leaves': 30,
    'feature_fraction': 0.9,
    'subsample': 0.8,
    'min_data_in_leaf': 100,
    'lambda_l2': 4,
    'objective': 'regression_l2', 
    'zero_as_missing': True,
    'metric': 'mae',
    'seed': 2}
    


train_set = lgb.Dataset(X_train, y_train, silent=False)
valid_set = lgb.Dataset(X_test, y_test, silent=False)
model = lgb.train(params, train_set = train_set, num_boost_round=10000,early_stopping_rounds=100,verbose_eval=500, valid_sets=valid_set)


test = pd.read_csv('../input/test_V2.csv')

test.drop(['Id','matchType','rankPoints'], axis=1, inplace=True)

test  = featureEngineering(test)

test.drop(['matchId','groupId'], axis=1, inplace=True)
test  = test[train_columns_new]

preds = model.predict(test, num_iteration = model.best_iteration)
preds[preds > 1] = 1

test  = pd.read_csv('../input/test_V2.csv')
test['winPlacePercPred'] = preds
aux = test.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
aux.columns = ['matchId','groupId','winPlacePerc']
test = test.merge(aux, how='left', on=['matchId','groupId'])
subm = test[['Id','winPlacePerc']]
subm.to_csv("lgb_baseline.csv", index=False)
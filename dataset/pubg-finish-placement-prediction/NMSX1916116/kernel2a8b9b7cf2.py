#%%
import os, sys, gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

import time

from sklearn.model_selection import KFold
import lightgbm as lgb

# from itertools import product

# %matplotlib inline

#%%
# static assets
trainFile = '../input/pubg-finish-placement-prediction/train_V2.csv'
testFile = '../input/pubg-finish-placement-prediction/test_V2.csv'
submitFile = './submission.csv'

level={\
    0 : [''],\
    1 : ['matchId'],\
    2 : ['matchId','groupId']
}

counting = {
    0: 'total_dataset', \
    1: 'total_id_in_match', \
    2: 'total_id_in_group'
}

NULL_T = 0x0000
MINN_T = 0x0001
MEAN_T = 0x0010
MAXX_T = 0x0100
RANK_T = 0x1000

suffix={
    MINN_T : ['_MINN', np.min],\
    MEAN_T : ['_MEAN', np.mean],\
    MAXX_T : ['_MAXX', np.max],\
    RANK_T : ['_RANK', None]
}

#%%
# loading data
def loadData(filepath, t_nrows=None):
    return pd.DataFrame(pd.read_csv(filepath, nrows=t_nrows))


#%%
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
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

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


#%%
# pre-processing features
def pre_process(trainingSet=True, debug=False):
    X, Y, Cols, Idx = None, None, None, None

    if trainingSet:
        print('loading trainning data ...')
        if debug:
            df = loadData(filepath=trainFile, t_nrows=30000)
        else:
            df = loadData(filepath=trainFile)
        df.dropna(axis=0, inplace=True)
    else:
        print('loading testing data ...')
        df = loadData(filepath=testFile)
        Idx = df['Id'].copy()
    
    df['totalDistance'] = df['rideDistance'] + df['walkDistance'] + df['swimDistance']
    df['healBoosts'] = df['heals'] + df['boosts']
    df['DBNOs2'] = df['vehicleDestroys'] + df['DBNOs']
    df['headshotRate'] = df['headshotKills'] / df['kills']
    df['headshotRate'] = df['headshotRate'].fillna(0)

    features=list(df.columns)
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchType")
    
    if trainingSet:
        Y=df.groupby(by=level[2])['winPlacePerc'].agg(suffix[MEAN_T][1]).astype(np.float64)
        X=Y.reset_index()[level[2]]
        df.drop(['winPlacePerc'],axis=1,inplace=True)
        features.remove("winPlacePerc")
        print('Normal trainning set ...')
    else:
        X=df[level[2]]
        print('Normal testing set ...')
    # group
    df_t = df.groupby(level[2]).size().reset_index(name=counting[2])
    X = X.merge(reduce_mem_usage(df_t), how='left', on=level[2])
    
    for v in tqdm(iterable=[MINN_T,MEAN_T,MAXX_T],desc='normalizing group',mininterval=4):
        df_t=df.groupby(by=level[2])[features].agg(suffix[v][1])
        df_t_rank=df_t.groupby(by=level[1])[features].rank(pct=True).reset_index()
        X=X.merge(
            reduce_mem_usage(df_t.reset_index()),
            suffixes=["", ""],
            how='left',
            on=level[2])
        X=X.merge(
            reduce_mem_usage(df_t_rank),
            suffixes=[suffix[v][0]+'_Group', suffix[v][0]+suffix[RANK_T][0]+'_Group'],
            how='left',
            on=level[2])
    
    # match
    df_t = df.groupby(level[1]).size().reset_index(name=counting[1])
    X = X.merge(reduce_mem_usage(df_t), how='left', on=level[1])
    
    df_rank=df.groupby(by=level[1])[features].rank(pct=True)
    X = X.join(reduce_mem_usage(df_rank), how='left')
    
    for v in tqdm(iterable=[MINN_T,MEAN_T,MAXX_T],desc='normalizing match',mininterval=4):
        df_t=df.groupby(by=level[1])[features].agg(suffix[v][1]).reset_index()
        X=X.merge(
            reduce_mem_usage(df_t),
            suffixes=['', suffix[v][0]+'_Match'],
            how='left',
            on=level[1])

    X.drop(level[2], axis=1, inplace=True)

    if trainingSet:
        Cols = X.columns

    del df, df_t, df_rank
    gc.collect()

    return X, Y, Cols, Idx


#%%
# load data set
x_train, y_train, c_features, _ = pre_process(trainingSet=True, debug=False)
x_test, _, _, i_test = pre_process(trainingSet=False, debug=False)

#%%
# LightGBM
folds = KFold(n_splits=3, shuffle=False, random_state=None)
trn_predict = np.zeros(x_train.shape[0])
sub_predict = np.zeros(x_test.shape[0])

t_st = time.process_time()
for t_fold, (t_trn, t_val) in enumerate(folds.split(x_train, y_train)):
    x_trn, y_trn = x_train.iloc[t_trn], y_train[t_trn]
    x_val, y_val = x_train.iloc[t_val], y_train[t_val]

    t_data_trn = lgb.Dataset(data=x_trn, label=y_trn)
    t_data_val = lgb.Dataset(data=x_val, label=y_val)
    t_params = {
        'objective': 'regression',
        'metric': 'mae',
        'n_estimators': 20000,
        'early_stopping_rounds': 200,
        # "num_leaves" : 31,
        'learning_rate': 0.05,
        'num_threads': 4,
        'colsample_bytree': 0.6,
        'bagging_fraction': 0.8,
        'bagging_seed': 1
    }
    reg = lgb.train(params=t_params,
                    train_set=t_data_trn,
                    valid_sets=[t_data_trn, t_data_val],
                    verbose_eval=1000)

    trn_predict[t_val] = reg.predict(x_val, num_iteration=reg.best_iteration)
    trn_predict = np.clip(a=trn_predict, a_min=0.0, a_max=1.0)
    
    t_sub_pred = reg.predict(x_test, num_iteration=reg.best_iteration)
    t_sub_pred = np.clip(a=t_sub_pred, a_min=0.0, a_max=1.0)
    sub_predict += t_sub_pred / folds.n_splits
    
    gc.collect()

t_en = time.process_time()

t_diff = t_en - t_st
print('Time: {:.2f} s'.format(t_diff))

#%%
sub_predict = np.clip(a=sub_predict, a_min=0.0, a_max=1.0)

#%%
# Align with maxPlace
# Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4
df_t = loadData(filepath=testFile)
for i in tqdm(range(len(df_t)), desc='fixing ans ...', mininterval=10):
    ans = sub_predict[i]
    maxPlace = int(df_t.iloc[i]['maxPlace'])
    if maxPlace == 0:
        ans = 0.0
    elif maxPlace == 1.0:
        ans = 1.0
    else:
        gap = 1.0 / (maxPlace - 1)
        ans = np.around(ans / gap) * gap
    if ans < 0:
        ans=0.0
    if ans > 1:
        ans=1.0
    sub_predict[i] = ans

df_t['winPlacePerc'] = sub_predict

#%%
df_sub = df_t[['Id', 'winPlacePerc']]

#%%
df_sub.to_csv(submitFile, index=False)
print('finish')

#%%

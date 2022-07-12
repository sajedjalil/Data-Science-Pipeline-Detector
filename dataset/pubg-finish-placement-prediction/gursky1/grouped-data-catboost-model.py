#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 19:27:06 2018

@author: Jacob
"""

#Importing libraries
import pandas as pd
import os
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import skew
import matplotlib
import matplotlib.pyplot as plt
import warnings
from scipy.special import boxcox1p
warnings.filterwarnings("ignore")

#######################################
#Data Import and Cleaning
#######################################
#Importing data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Memory Optimization
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
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
        elif 'datetime' not in col_type.name:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

#Running optimization
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

#We can get rid of the Id column for both
train_id = train['Id']
train_match = train['matchId']
train_group = train['groupId']
test_id = test['Id']
test_match = test['matchId']
test_group = test['groupId']

#Lets concat our datasets
ntrain = train.shape[0]
ntest = test.shape[0]
grouped_y = train[['groupId','winPlacePerc']]
y_train = train.winPlacePerc.values
train['train'] = 1
test['train'] = 0
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['winPlacePerc'], axis=1, inplace=True)

###############################################################################
#Feature Engineering
###############################################################################

#Lets create a variable that denotes team size
team_sizes = all_data.groupby(['matchId','groupId'])['Id'].agg(['count'])
team_sizes.columns = ['teamSize']
all_data = all_data.merge(team_sizes, 
                          left_on = ['matchId','groupId'], 
                          right_index = True, 
                          how = 'left')
all_data['solo'] = np.where(all_data['teamSize']==1, 1, 0)

#Lets create a variable that compares much better the player is than the average KillPoints
avg_kp = all_data.groupby(['matchId'])['killPoints','winPoints'].agg(['mean'])
avg_kp.columns = ['matchAvgKills','matchAvgWins']
all_data = all_data.merge(avg_kp, 
                          left_on = ['matchId'], 
                          right_index = True, 
                          how = 'left')
all_data['killDiff'] = all_data['killPoints'] - all_data['matchAvgKills']
all_data['winDiff'] = all_data['winPoints'] - all_data['matchAvgWins']

#Lets create a variable that measures the group's average external stats
group_avg_kp = all_data.groupby(['matchId','groupId'])['killPoints','winPoints'].agg(['mean'])
group_avg_kp.columns = ['groupAvgKills','groupAvgWins']
all_data = all_data.merge(group_avg_kp, 
                          left_on = ['matchId','groupId'], 
                          right_index = True, 
                          how = 'left')

#Lets create an indicator for when walkDistance is 0
all_data['noWalk'] = np.where(all_data['walkDistance']==0,1,0)

#Lets do the same thing for swimming and driving
all_data['noSwim'] = np.where(all_data['swimDistance']==0,1,0)
all_data['noRide'] = np.where(all_data['rideDistance']==0,1,0)

#Lets create some more variables from a kernel
all_data["distance"] = all_data["rideDistance"]+all_data["walkDistance"]+all_data["swimDistance"]
all_data["healthpack"] = all_data["boosts"] + all_data["heals"]
all_data["skill"] = all_data["headshotKills"]+all_data["roadKills"]

#Lets set our Id level predictors so we can group by all others
dont_use_feats = ['Id','matchId','groupId','numGroups','matchAvgKills','matchAvgWins',
                  'teamSize','groupAvgKills','groupAvgWins','train']

#Grouping by team
agg_team = {c: ['mean', 'min', 'max', 'sum'] for c in [c for c in all_data.columns if c not in dont_use_feats]}
agg_team['numGroups'] = ['size']
#print(agg_team.keys())

def preprocess(df):    
    df_gb = df.groupby('groupId').agg(agg_team)
    df_gb.columns = pd.Index([e[0] + "_" + e[1].upper() for e in df_gb.columns])    
    return df_gb

all_data = preprocess(all_data)
#this is needed, since for some teams sum of rideDistance is infinite. This is not swallowed by LIME
all_data = all_data.replace({np.inf: -1})

y_train = grouped_y.groupby('groupId')['winPlacePerc'].median()
# since we train on the group and out final metric is on user level, we want to assign group size as the weight
w = all_data['numGroups_SIZE']

#Lets drop some columns from our training set that are probably not helpful
#all_data_backup = all_data
#all_data = all_data.drop(['Id','matchId','groupId'], axis = 1)

#Turning all_data into train and test
#X_train = all_data[all_data['train']==1].drop(['train'], axis = 1)
#X_test = all_data[all_data['train']==0].drop(['train'], axis = 1)

#Lets make a reduced data set
X_train = all_data.loc[y_train.index.tolist()]
X_test = all_data.loc[~all_data.index.isin(y_train.index.tolist())]
X_train = X_train.sample(frac = 0.25, random_state = 123)
y_train = y_train[X_train.index]
w_train = w[X_train.index]

###############################################################################
#Lets remove some objects we don't need
###############################################################################
del all_data, avg_kp, group_avg_kp, team_sizes

###############################################################################
#Modeling!
###############################################################################

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Lasso, Ridge, RidgeCV
#from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
#from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, minmax_scale
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from mlxtend.regressor import StackingRegressor, StackingCVRegressor
from sklearn.feature_selection import RFECV
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from numpy import inf

#Train/valid split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=123)


#Lets do a CatBoost
cat = CatBoostRegressor(iterations = 3000,
                        learning_rate=0.05,
                        depth=12,
                        eval_metric='MAE',
                        random_seed = 123,
                        od_type='Iter',
                        metric_period = 50,
                        od_wait=20,
                        thread_count = 3,
                        colsample_bylevel = 0.8,
                        verbose = True)
cat.fit(X_train, 
        y_train,
        eval_set=(X_valid,y_valid),
        use_best_model=True)

#Submission of Scores
sub = pd.read_csv('../input/sample_submission.csv')
sub['Id'] = test_id
sub['matchId']=test_match
sub['groupId']=test_group
pred = pd.Series(np.clip(cat.predict(X_test),0,1), index = X_test.index.tolist()).to_frame()
pred.columns = ['pred']
sub = sub.merge(pred, how = 'left', left_on = "groupId", right_index = True)
sub.drop(['winPlacePerc'], axis =1, inplace = True)
sub.columns = ['Id','matchId','groupId','pred']
subg = sub.groupby(['matchId','groupId'])['pred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
subg.columns = ['matchId','groupId','winPlacePerc']
sub = sub.merge(subg, how='left', on=['matchId','groupId'])
sub = sub[['Id','matchId','groupId','winPlacePerc']]
sub['winPlacePerc'] = sub.groupby('matchId')['winPlacePerc'].transform(lambda x: minmax_scale(x.values.astype(float)))
sub = sub[['Id','winPlacePerc']]
sub.to_csv('submission.csv', index=False)
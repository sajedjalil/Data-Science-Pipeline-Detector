# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# -*- coding: utf-8 -*-

##################################################################################################################
#                                                 MACHINE LEARNING                                               # 
##################################################################################################################

#Importing the functions
import pandas as pd
import numpy as np
import gc
import lightgbm as lgbm
import os
gc.enable()
print(os.listdir("../input"))

#Gathering Data
print("Gathering data...")
training_data = pd.read_csv("/kaggle/input/pubg-finish-placement-prediction/train_V2.csv") #Reading the training file
testing_data = pd.read_csv("/kaggle/input/pubg-finish-placement-prediction/test_V2.csv") #Reading the testing file
print("Done \n")
 
#Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(data):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = data.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in data.columns:
        col_type = data[col].dtype

        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

    end_mem = data.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return data

#Dropping irrevelent features from both sets
print ("Dropping irrevelant features...")
training_data.drop(columns = ['Id', 'boosts', 'damageDealt', 'DBNOs', 'killPlace', 'killPoints',
                    'longestKill', 'matchDuration','maxPlace', 'numGroups', 'revives', 'roadKills', 
                    'vehicleDestroys', 'weaponsAcquired', 'matchType', 'rankPoints'], inplace = True) 
    
testing_data.drop(columns = ['Id', 'boosts', 'damageDealt', 'DBNOs', 'killPlace', 'killPoints',
                    'longestKill', 'matchDuration','maxPlace', 'numGroups', 'revives', 'roadKills', 
                    'vehicleDestroys', 'weaponsAcquired', 'matchType','rankPoints'], inplace = True) 
print("Done \n")
   
#Dropping the lignes where stats are not completed for all columns 
print("Dropping missing values...")    
training_data = training_data.dropna(how='any',axis=0) 
print("Done \n")

#Creating a list of features that will be used to compare the stats
features = list(training_data.columns) 
features.remove("matchId")
features.remove("groupId")
features.remove("winPlacePerc")

#Giving the mean of each feature per game
print("Getting match mean feature...") 
match_mean = training_data.groupby(['matchId'])[features].agg('mean').reset_index()
training_data = training_data.merge(match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
match_mean = testing_data.groupby(['matchId'])[features].agg('mean').reset_index()
testing_data = testing_data.merge(match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
print("Done \n")

#Counting the number of players per game
print("Getting match size feature...") 
match_size = training_data.groupby(['matchId']).size().reset_index(name='match_size')
training_data = training_data.merge(match_size, how='left', on=['matchId'])
match_size = testing_data.groupby(['matchId']).size().reset_index(name='match_size')
testing_data = testing_data.merge(match_size, how='left', on=['matchId'])
print("Done \n")

#Counting the number of teams per game
print("Getting group size feature...") 
group_size = training_data.groupby(['matchId','groupId']).size().reset_index(name='group_size')
training_data = training_data.merge(group_size, how='left', on=['matchId', 'groupId'])
group_size = testing_data.groupby(['matchId','groupId']).size().reset_index(name='group_size')
testing_data = testing_data.merge(group_size, how='left', on=['matchId', 'groupId'])
print("Done \n")

print("Getting group max feature...")
group_max = training_data.groupby(['matchId','groupId'])[features].agg('max')
group_max_rank = group_max.groupby('matchId')[features].rank(pct=True).reset_index()
training_data = training_data.merge(group_max.reset_index(), suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
training_data = training_data.merge(group_max_rank, suffixes=["", "_max_rank"], how='left', on=['matchId', 'groupId'])
group_max = testing_data.groupby(['matchId','groupId'])[features].agg('max')
group_max_rank = group_max.groupby('matchId')[features].rank(pct=True).reset_index()
testing_data = testing_data.merge(group_max.reset_index(), suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
testing_data = testing_data.merge(group_max_rank, suffixes=["", "_max_rank"], how='left', on=['matchId', 'groupId'])
print("Done \n")

print("Getting group min feature...")
group_min = training_data.groupby(['matchId','groupId'])[features].agg('min')
group_min_rank = group_min.groupby('matchId')[features].rank(pct=True).reset_index()
training_data = training_data.merge(group_min.reset_index(), suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
training_data = training_data.merge(group_min_rank, suffixes=["", "_min_rank"], how='left', on=['matchId', 'groupId'])
group_min = testing_data.groupby(['matchId','groupId'])[features].agg('min')
group_min_rank = group_min.groupby('matchId')[features].rank(pct=True).reset_index()
testing_data = testing_data.merge(group_min.reset_index(), suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
testing_data = testing_data.merge(group_min_rank, suffixes=["", "_min_rank"], how='left', on=['matchId', 'groupId'])
print("Done \n")

#Calling the function to reduce the memory
print("Reducing memory...")
training_data = reduce_mem_usage(training_data)
print('_'*40)
testing_data = reduce_mem_usage(testing_data)
print("Done \n")

#Dropping features that are judged irrevelent for the program
print ("Dropping irrevelant features...")
training_data.drop(["matchId", "groupId"], axis=1, inplace=True)
testing_data.drop(["matchId", "groupId"], axis=1, inplace=True)
print("Done \n")

#Deleting and collecting in the garbage
del match_mean, match_size, group_size, group_max, group_max_rank, group_min, group_min_rank
gc.collect()

#Creating new relevant features
print("Creating totalDistance...")
training_data['totalDistance'] = training_data['rideDistance'] + training_data['swimDistance'] + training_data['walkDistance']
testing_data['totalDistance'] = testing_data['rideDistance'] + testing_data['swimDistance'] + testing_data['walkDistance']
print("Done \n")

print("Creating ratio_kills_walkDistance...")
training_data['ratio_kills_walkDistance'] = training_data['kills'] / training_data['walkDistance']
training_data['ratio_kills_walkDistance'].fillna(0, inplace=True)
training_data['ratio_kills_walkDistance'].replace(np.inf, 0, inplace=True)
testing_data['ratio_kills_walkDistance'] = testing_data['kills'] / testing_data['walkDistance']
testing_data['ratio_kills_walkDistance'].fillna(0, inplace=True)
testing_data['ratio_kills_walkDistance'].replace(np.inf, 0, inplace=True)
print("Done \n")

print("Creating ratio_headshotKills_kills...")
training_data['ratio_headshotKills_kills'] = training_data['headshotKills'] / training_data['kills']
training_data['ratio_headshotKills_kills'].fillna(0, inplace=True)
training_data['ratio_headshotKills_kills'].replace(np.inf, 0, inplace=True)
testing_data['ratio_headshotKills_kills'] = testing_data['headshotKills'] / testing_data['kills']
testing_data['ratio_headshotKills_kills'].fillna(0, inplace=True)
testing_data['ratio_headshotKills_kills'].replace(np.inf, 0, inplace=True)
print("Done \n")

##################################################################################################################
#                                                   MODELING                                                     # 
##################################################################################################################

#Creating matrices with truncated values for the training features
x_test = np.asarray(testing_data)
training_data_truncated = training_data.truncate(after=1000000)
x_train_truncated = np.asarray(training_data_truncated.drop(['winPlacePerc'], axis = 1))
y_train_truncated = np.asarray(training_data_truncated[['winPlacePerc']])

#Deleting and collecting in the garbage
del training_data, training_data_truncated, testing_data
gc.collect()

#Predicting the results of the testing set with the model, credit to https://www.kaggle.com/brijrokad/lgbmregression-in-pubg
print("Modeling...")
prediction = lgbm.LGBMRegressor(learning_rate=0.05, bagging_fraction = 0.9, max_bin = 127, metric = 'mae',
                                n_estimators = 1000, n_jobs=-1,boosting_type = 'gbdt', max_depth = 30, min_data_in_leaf = 10,
                                num_leaves = 200).fit(x_train_truncated, y_train_truncated).predict(x_test)

#Submitting the prediction to a new CSV file
testing_data = pd.read_csv("/kaggle/input/pubg-finish-placement-prediction/test_V2.csv")
submission = testing_data.copy()
submission['winPlacePerc'] = prediction
submission.to_csv('submission.csv', columns=['Id', 'winPlacePerc'], index=False)
submission[['Id', 'winPlacePerc']].head()
#---------------------------------------------------------------------------------------------------------
# Data analysis and wrangling
import numpy as np
import pandas as pd

# Machine learning
import sklearn as skl
import lightgbm as lgb
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error

# File handling
import os
import gc
gc.enable()
print(os.listdir("../input"))

#Gather Data
training_df = pd.read_csv("../input/train_V2.csv")
testing_df = pd.read_csv("../input/test_V2.csv")
print("Data loaded")
#---------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

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

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
#---------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------
#Dropping irrelevant features 
training_df = training_df.drop(['Id', 'longestKill', 'rankPoints', 'numGroups', 'matchType'], axis=1)
testing_df = testing_df.drop(['Id', 'longestKill', 'rankPoints', 'numGroups', 'matchType'], axis=1)
print("Irrelevant features dropped")

#Completing missing values
training_df = training_df.dropna(how='any',axis=0)
print("Missing values completed")

features = list(training_df.columns)
features.remove("matchId")
features.remove("groupId")
features.remove("winPlacePerc")

print("get match mean feature")
match_mean = training_df.groupby(['matchId'])[features].agg('mean').reset_index()
training_df = training_df.merge(match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
match_mean = testing_df.groupby(['matchId'])[features].agg('mean').reset_index()
testing_df = testing_df.merge(match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])

print("get match size feature")
match_size = training_df.groupby(['matchId']).size().reset_index(name='match_size')
training_df = training_df.merge(match_size, how='left', on=['matchId'])
match_size = testing_df.groupby(['matchId']).size().reset_index(name='match_size')
testing_df = testing_df.merge(match_size, how='left', on=['matchId'])

print("get group size feature")
group_size = training_df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
training_df = training_df.merge(group_size, how='left', on=['matchId', 'groupId'])
group_size = testing_df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
testing_df = testing_df.merge(group_size, how='left', on=['matchId', 'groupId'])

print("get group max feature")
group_max = training_df.groupby(['matchId','groupId'])[features].agg('max')
group_max_rank = group_max.groupby('matchId')[features].rank(pct=True).reset_index()
training_df = training_df.merge(group_max.reset_index(), suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
training_df = training_df.merge(group_max_rank, suffixes=["", "_max_rank"], how='left', on=['matchId', 'groupId'])
group_max = testing_df.groupby(['matchId','groupId'])[features].agg('max')
group_max_rank = group_max.groupby('matchId')[features].rank(pct=True).reset_index()
testing_df = testing_df.merge(group_max.reset_index(), suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
testing_df = testing_df.merge(group_max_rank, suffixes=["", "_max_rank"], how='left', on=['matchId', 'groupId'])

print("get group min feature")
group_min = training_df.groupby(['matchId','groupId'])[features].agg('min')
group_min_rank = group_min.groupby('matchId')[features].rank(pct=True).reset_index()
training_df = training_df.merge(group_min.reset_index(), suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
training_df = training_df.merge(group_min_rank, suffixes=["", "_min_rank"], how='left', on=['matchId', 'groupId'])
group_min = testing_df.groupby(['matchId','groupId'])[features].agg('min')
group_min_rank = group_min.groupby('matchId')[features].rank(pct=True).reset_index()
testing_df = testing_df.merge(group_min.reset_index(), suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
testing_df = testing_df.merge(group_min_rank, suffixes=["", "_min_rank"], how='left', on=['matchId', 'groupId'])

training_df = reduce_mem_usage(training_df)
print('_'*40)
testing_df = reduce_mem_usage(testing_df)
print("Memory reduced")

print("get group mean feature")
group_mean = training_df.groupby(['matchId','groupId'])[features].agg('mean')
group_mean_rank = group_mean.groupby('matchId')[features].rank(pct=True).reset_index()
training_df = training_df.merge(group_mean.reset_index(), suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
training_df = training_df.merge(group_mean_rank, suffixes=["", "_mean_rank"], how='left', on=['matchId', 'groupId'])
group_mean = testing_df.groupby(['matchId','groupId'])[features].agg('mean')
group_mean_rank = group_mean.groupby('matchId')[features].rank(pct=True).reset_index()
testing_df = testing_df.merge(group_mean.reset_index(), suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
testing_df = testing_df.merge(group_mean_rank, suffixes=["", "_mean_rank"], how='left', on=['matchId', 'groupId'])

training_df.drop(["matchId", "groupId"], axis=1, inplace=True)
testing_df.drop(["matchId", "groupId"], axis=1, inplace=True)
print("Irrelevant features dropped")

training_df = reduce_mem_usage(training_df)
print('_'*40)
testing_df = reduce_mem_usage(testing_df)
print("Memory reduced")

del match_mean, match_size, group_size, group_max, group_max_rank, group_min, group_min_rank, group_mean, group_mean_rank
gc.collect()
#---------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------
#Creating new potentially relevant values

print("Create headshotRate")
training_df['headshotRate'] = training_df['headshotKills'] / training_df['kills']
training_df['headshotRate'].fillna(0, inplace=True)
training_df['headshotRate'].replace(np.inf, 0, inplace=True)
testing_df['headshotRate'] = testing_df['headshotKills'] / training_df['kills']
testing_df['headshotRate'].fillna(0, inplace=True)
testing_df['headshotRate'].replace(np.inf, 0, inplace=True)

print("Create totalDistance")
training_df['totalDistance'] = training_df['rideDistance'] + training_df['swimDistance'] + training_df['walkDistance']
testing_df['totalDistance'] = testing_df['rideDistance'] + testing_df['swimDistance'] + testing_df['walkDistance']

print("Create items")
training_df['items'] = training_df['heals'] + training_df['boosts']
testing_df['items'] = testing_df['heals'] + testing_df['boosts']

print("Create healsPerWalkDistance")
training_df['healsPerWalkDistance'] = training_df['heals'] / training_df['walkDistance']
training_df['healsPerWalkDistance'].fillna(0, inplace=True)
training_df['healsPerWalkDistance'].replace(np.inf, 0, inplace=True)
testing_df['healsPerWalkDistance'] = testing_df['heals'] / testing_df['walkDistance']
testing_df['healsPerWalkDistance'].fillna(0, inplace=True)
testing_df['healsPerWalkDistance'].replace(np.inf, 0, inplace=True)

print("Create killsPerWalkDistance")
training_df['killsPerWalkDistance'] = training_df['kills'] / training_df['walkDistance']
training_df['killsPerWalkDistance'].fillna(0, inplace=True)
training_df['killsPerWalkDistance'].replace(np.inf, 0, inplace=True)
testing_df['killsPerWalkDistance'] = testing_df['kills'] / testing_df['walkDistance']
testing_df['killsPerWalkDistance'].fillna(0, inplace=True)
testing_df['killsPerWalkDistance'].replace(np.inf, 0, inplace=True)
#---------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------
#Modeling

#Creating matrices
training_df_truncated = training_df.truncate(after=2000000)
X_train_truncated = np.asarray(training_df_truncated.drop(['winPlacePerc'], axis = 1))
y_train_truncated = np.asarray(training_df_truncated[['winPlacePerc']])
#X_train = np.asarray(training_df.drop(['winPlacePerc'], axis = 1))
#y_train = np.asarray(training_df[['winPlacePerc']])
X_test = np.asarray(testing_df)

#print ('Train set:', X_train.shape,y_train.shape)
print ('Test set:', X_test.shape)

del training_df, training_df_truncated, testing_df
gc.collect()

# Predicting the results of the testing set with the model
print("Modeling...")
yhat_test = lgb.LGBMRegressor(learning_rate=0.05, bagging_fraction = 0.9, max_bin = 127, metric = 'mae', n_estimators = 1000, n_jobs=-1,boosting_type = 'gbdt', max_depth = 30, min_data_in_leaf = 10, num_leaves = 200).fit(X_train_truncated, y_train_truncated).predict(X_test)

# Submitting
testing_df = pd.read_csv("../input/test_V2.csv")
submission = testing_df.copy()
submission['winPlacePerc'] = yhat_test
submission.to_csv('submission.csv', columns=['Id', 'winPlacePerc'], index=False)
submission[['Id', 'winPlacePerc']].head()
#---------------------------------------------------------------------------------------------------------
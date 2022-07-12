
# coding: utf-8

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import gc
import seaborn as sns
from matplotlib import pyplot as plt
# from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import warnings
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import ShuffleSplit
import os, sys
from pprint import pprint
import time


print(f"Start... {time.strftime('%H:%M')}")

tr = pd.read_csv('../input/train_V2.csv')
ts = pd.read_csv('../input/test_V2.csv')

gc.enable()
pd.set_option("max_columns", 200)
pd.set_option("max_rows", 200)

# drop na rows
tr.dropna(axis=0, inplace=True)
tr.reset_index(drop=True, inplace=True)


# get a subset of the training data (tr) after sorting it by matchId
# and ensure that all rows of a given match are included in the subset
tr_limit = int(0.5 * tr.shape[0]) # the fraction to keep of tr
tr.sort_values(by='matchId', inplace=True); tr.reset_index(drop=True, inplace=True)
tr = tr.iloc[0:tr_limit+1, :]
mId = tr.at[tr_limit, 'matchId']
for i in range(tr_limit, tr_limit-200, -1):
    if tr.at[i, 'matchId'] == mId:
        tr.at[i, 'matchId'] = np.nan
tr.dropna(axis=0, inplace=True); tr.reset_index(drop=True, inplace=True)

tt = [tr, ts]


# feature engineering
for i, df in enumerate(tt):
    df.drop(['rankPoints', 'killPoints', 'winPoints', 'maxPlace', 'matchDuration'], axis=1, inplace=True)

    # adding memberCount column
    map_df = df.groupby(["matchId", "groupId"], sort=False, as_index=False)[["Id"]].count()
    map_df["MatchGroupId"] = map_df["matchId"] + map_df["groupId"]
    map_df = map_df[["MatchGroupId", "Id"]]
    map_dict = {}
    for k, v in zip(list(map_df['MatchGroupId']), list(map_df['Id'])):
        map_dict[k] = v
    df['memberCount'] = (df['matchId'] + df['groupId']).map(map_dict)
    del map_df; gc.collect()

    # adding polynomial features
    poly_feat = ['walkDistance', 'killPlace', 'damageDealt', 'kills']
    for f in poly_feat:
        df[f+'_2'] = df[f] ** 2
        df[f+'_sqrt'] = np.sqrt(df[f]) 
    for f in ['killPlace', 'walkDistance']:
        df[f+'_3'] = df[f] ** 3

    df['killPlace_exp'] = np.power(np.e, df['killPlace'])
    df['assists_revives__teamKills'] = df['assists'] + df['revives'] + 2 / (df['teamKills'] + 1)
    df['distanceSum'] = df['walkDistance'] + df['swimDistance'] + df['rideDistance']
    df['heals_boosts'] = df['heals'] + df['boosts']
    df['longestKill_headshotKills_killStreaks'] = (df['longestKill']+1) * (df['headshotKills'] + df['killStreaks'])
    df['DBNOs_kills'] = df['DBNOs'] + df['kills']

    # adding group data
    matchGr = df.drop(['Id', 'matchType', 'numGroups', 'groupId', 'memberCount'], 
                      axis=1).groupby('matchId').agg(np.sum)
    matchGroupGr = df.drop(['Id', 'matchType', 'numGroups', 'memberCount'], 
                           axis=1).groupby(['matchId', 'groupId']).agg(np.sum)
    grpCols = list(set(df.columns.values) - 
                   set(['Id', 'groupId', 'matchId', 'matchType', 
                        'numGroups', 'winPlacePerc', 'memberCount']))

    for col in grpCols:
        df['Group__' + col] = 0.000000000000001
        
    for col in grpCols:
        for i in range(df.shape[0]):
            matchVal = matchGr.at[df.at[i, 'matchId'], col]
            groupVal = matchGroupGr.at[(df.at[i, 'matchId'], df.at[i, 'groupId']), col]
            # to avoid division by 0
            if matchVal == 0:
                df.at[i, 'Group__' + col] = 0.0
            else:
                df.at[i, 'Group__' + col] = 10 * groupVal / matchVal
    del matchGr, matchGroupGr

    ## adding a column to represent the percentage of group members in total match players
    matchGr = df[['matchId', 'Id']].groupby('matchId').count()
    matchGroupGr = df[['matchId', 'groupId', 'Id']].groupby(['matchId', 'groupId']).count()
    df['Group__memberCount'] = 0
    for i in range(df.shape[0]):
        matchVal = matchGr.at[df.at[i, 'matchId'], 'Id']
        groupVal = matchGroupGr.at[(df.at[i, 'matchId'], df.at[i, 'groupId']), 'Id']
        df.at[i, 'Group__memberCount'] = groupVal / matchVal
    del matchGr, matchGroupGr
    gc.collect()


# Dropping id columns and creating dummy variables
test_id = ts["Id"]
for df in tt:
    df.drop(columns=['Id', 'groupId', 'matchId'], axis=1, inplace=True)

tr = pd.get_dummies(tr, columns=['matchType']) 
ts = pd.get_dummies(ts, columns=['matchType']) 


X_train, X_test, y_train, y_test = train_test_split(tr.drop('winPlacePerc', axis=1), 
                                                    tr['winPlacePerc'], test_size=0.25, 
                                                    random_state=3)
colNames = tr.columns.values; del tr, X_test, y_test; gc.collect()


# Modeling
xgb = XGBRegressor(random_state=3, n_jobs=1, subsample=0.7, reg_lambda=50, 
                    reg_alpha=10, n_estimators=360, learning_rate=0.8, 
                    gamma=0, colsample_bytree=1, colsample_bylevel=0.7)

model = xgb
model.fit(X_train, y_train)
ts_pred = model.predict(ts)

# Post-processing
ts_pred = [0 if p < 0 else p for p in ts_pred]
ts_pred = [1 if p > 1 else p for p in ts_pred]
ts_numGroups = ts['numGroups'].reset_index(drop=True)
for i in (range(ts.shape[0])):
    numGroups = ts_numGroups[i]
    winPerc = ts_pred[i]
    scores_ = np.around(np.linspace(0, 1, numGroups), decimals=4)
    score_diff = np.abs(scores_ - winPerc)
    ts_pred[i] = scores_[np.argmin(score_diff)]

# submission
submission = pd.DataFrame({
    'Id': test_id,
    'winPlacePerc': ts_pred
})
submission.to_csv('submission.csv', index=False)

print(f"End. {time.strftime('%H:%M')}")

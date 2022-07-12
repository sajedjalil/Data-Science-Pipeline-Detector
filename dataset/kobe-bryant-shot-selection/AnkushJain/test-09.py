# coding: utf-8
__author__ = 'Ravi: https://kaggle.com/company'

import numpy as np
import pandas as pd

from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import cross_val_score


import math
from functools import reduce

# Load data and roughly clean it, then sort as game date
df = pd.read_csv("../input/data.csv")
df['goals_in_game']=0
df['last_goal']=0
shotIds = df['shot_id']
game_id_prev = 0
for i in range(0, len(df)):
    if df.iloc[i]['game_id'] == game_id_prev :
	    lastGoal = df.iloc[i-1][ 'shot_made_flag']
	    if math.isnan(lastGoal ):
	        df.set_value(i,'goals_in_game',df.iloc[i-1][ 'goals_in_game'])
	        df.set_value(i,'last_goal', 0)
	    else:
	        df.set_value(i,'goals_in_game',df.iloc[i-1][ 'goals_in_game'] + lastGoal)
	        df.set_value(i,'last_goal',lastGoal)
    else:
	    df.set_value(i,'goals_in_game', 0)
	    df.set_value(i,'last_goal', 0)
	    game_id_prev = df.iloc[i]['game_id'] 
    


df.drop(['game_event_id', 'game_id', 'lat', 'lon', 'team_id', 'team_name'], axis=1, inplace=True)
df.sort_values('game_date',  inplace=True)
mask = df['shot_made_flag'].isnull()


actions1 = reduce( set.union, df.apply( lambda r: set(r['action_type'].lower().replace('-','').split(' ')  ), axis=1 ) )
drop_actions = set(["up", "oop", "roll"])
actions1 = actions1.difference(drop_actions)

for x in actions1 :
    df["action_"+x] = 0
    df["action_"+x] = df.apply( lambda r: r['action_type'].find(x)!=-1, axis=1)
    


# Clean data
#actiontypes = dict(df.action_type.value_counts())
#df['type'] = df.apply(lambda row: row['action_type'] if actiontypes[row['action_type']] > 20\
#                          else row['combined_shot_type'], axis=1)

df['away'] = df.matchup.str.contains('@')
df.drop('matchup', axis=1, inplace=True)

df['angle'] = df.apply(lambda row: math.atan2( row['loc_x'], row['loc_y']), axis=1 )

df['time_remaining'] = df.apply(lambda row: row['minutes_remaining'] * 60 + row['seconds_remaining'], axis=1)

df['year'] = df['game_date'].str[0:4].astype(int)
df['month'] =  df['game_date'].str[5:7].astype(int)
df['day'] =  df['game_date'].str[8:10].astype(int)
df['shot_type'] = df.apply(lambda row: row['shot_type'].find("2PT")!=-1, axis=1)

ctr = list(np.unique( df['opponent']))
df['opponent'] = df.apply( lambda x: ctr.index(x['opponent']) , axis=1)


# Need work on game_date, add this into feature and increse n_estimators can inprove results but waste time and memory 

df.drop(['action_type', 'combined_shot_type', 'loc_x', 'loc_y', 'season', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'game_date', 'shot_id', 'seconds_remaining', 'minutes_remaining' ], axis=1, inplace=True)

# Need work on game_date, add this into feature and increse n_estimators can inprove results but waste time and memory 

y = df.shot_made_flag[~mask]
df.drop(['shot_made_flag'],axis=1,inplace=True)
X = df[~mask]

print("-" * 10 + "XGBClassifier" + "-" * 10)
clf_xgb = XGBClassifier(max_depth=7, learning_rate=0.012, n_estimators=1000, subsample=0.62, colsample_bytree=0.6, seed=1)
clf_xgb.fit(X, y)

target_x = df[mask]
target_y = clf_xgb.predict_proba(target_x)[:,1]
target_id = shotIds[mask]
submission = pd.DataFrame({"shot_id":target_id, "shot_made_flag":target_y})
submission.sort_values('shot_id',  inplace=True)
submission.to_csv("file008.csv",index=False)

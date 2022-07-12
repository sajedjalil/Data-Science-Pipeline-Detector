# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def split_train_val(data, fraction):
    matchIds = data['matchId'].unique().reshape([-1])
    train_size = int(len(matchIds)*fraction)
    
    random_idx = np.random.RandomState(seed=2).permutation(len(matchIds))
    train_matchIds = matchIds[random_idx[:train_size]]
    val_matchIds = matchIds[random_idx[train_size:]]
    
    data_train = data.loc[data['matchId'].isin(train_matchIds)]
    data_val = data.loc[data['matchId'].isin(val_matchIds)]
    return data_train, data_val

train = pd.read_csv('../input/train_V2.csv')
data = train.copy()
data = data.sort_values("matchId")
data = data.reset_index(drop=True)
data.fillna(0, inplace=True)

data['killPerc'] = data.groupby('matchId')['kills'].rank(pct=True).values
data['killPlacePerc'] = data.groupby('matchId')['killPlace'].rank(pct=True).values
data['damageDealtPerc'] = data.groupby('matchId')['damageDealt'].rank(pct=True).values
data['distancePerc'] = data.groupby('matchId')['walkDistance'].rank(pct=True).values
data['boostsPerc'] = data.groupby('matchId')['boosts'].rank(pct=True).values
# data['revivesPerc'] = data.groupby('matchId')['revives'].rank(pct=True).values
data['headshotPerKill'] = data['headshotKills'] / data['killPerc'] 
data['killStreaksPerc'] = data.groupby('matchId')['killStreaks'].rank(pct=True).values
data['vehicleDestroysPerc'] = data.groupby('matchId')['vehicleDestroys'].rank(pct=True).values
data['swimDistancePerc'] = data.groupby('matchId')['swimDistance'].rank(pct=True).values
data['rideDistancePerc'] = data.groupby('matchId')['rideDistance'].rank(pct=True).values
data['killPoints'] = data.groupby('matchId')['killPoints'].rank(pct=True).values

data['killPerc_walkDistance'] = data['killPerc'] / data['walkDistance']
data['roadKillsPerc'] = data.groupby('matchId')['roadKills'].rank(pct=True).values

# data['_walkDistance_kills_Ratio6'] = all_data['walkDistance'] / data['killPerc']


features = ["matchId", "winPlacePerc", "longestKill", "distancePerc", "killPerc","boostsPerc", "damageDealtPerc", "killPlacePerc", "killStreaksPerc", 
            "swimDistancePerc", "rideDistancePerc", "vehicleDestroysPerc", "killPoints", "killPerc_walkDistance", "roadKillsPerc"]

data = data[features]

train_set, val_set = split_train_val(data, 0.9)

y_train = train_set['winPlacePerc']
y_val = val_set['winPlacePerc']

features.remove("matchId")
features.remove("winPlacePerc")

train_set = lgb.Dataset(train_set[features], label=y_train)
val_set = lgb.Dataset(val_set[features], label=y_val)



params = {
        "objective" : "regression", 
        "metric" : "mae", 
        "num_leaves" : 149, 
        "learning_rate" : 0.03, 
        "bagging_fraction" : 0.9,
        "bagging_seed" : 0, 
        "num_threads" : 30,
        "colsample_bytree" : 0.5,
        'min_data_in_leaf':1900, 
        'min_split_gain':0.00011,
        'lambda_l2':9
}


model = lgb.train(  params, 
                    train_set = train_set,
                    num_boost_round=9400,
                    early_stopping_rounds=200,
                    verbose_eval=100, 
                    valid_sets=[train_set,val_set]
                  )
                  
                  
test = pd.read_csv('../input/test_V2.csv')
data = test.copy()

data['killPerc'] = data.groupby('matchId')['kills'].rank(pct=True).values
data['killPlacePerc'] = data.groupby('matchId')['killPlace'].rank(pct=True).values
data['damageDealtPerc'] = data.groupby('matchId')['damageDealt'].rank(pct=True).values
data['distancePerc'] = data.groupby('matchId')['walkDistance'].rank(pct=True).values
data['boostsPerc'] = data.groupby('matchId')['boosts'].rank(pct=True).values
data['headshotPerKill'] = data['headshotKills'] / data['killPerc'] 
data['killStreaksPerc'] = data.groupby('matchId')['killStreaks'].rank(pct=True).values
data['vehicleDestroysPerc'] = data.groupby('matchId')['vehicleDestroys'].rank(pct=True).values
data['swimDistancePerc'] = data.groupby('matchId')['swimDistance'].rank(pct=True).values
data['rideDistancePerc'] = data.groupby('matchId')['rideDistance'].rank(pct=True).values
data['killPoints'] = data.groupby('matchId')['killPoints'].rank(pct=True).values

data['killPerc_walkDistance'] = data['killPerc'] / data['walkDistance']
data['roadKillsPerc'] = data.groupby('matchId')['roadKills'].rank(pct=True).values

# data['_walkDistance_kills_Ratio6'] = all_data['walkDistance'] / data['killPerc']

Ids = data["Id"].as_matrix()


features = ["longestKill", "distancePerc", "killPerc","boostsPerc", "damageDealtPerc", "killPlacePerc", "killStreaksPerc", 
            "swimDistancePerc", "rideDistancePerc", "vehicleDestroysPerc", "killPoints", "killPerc_walkDistance", "roadKillsPerc"]


test_set =data[features].as_matrix()

predictions = model.predict(test_set, num_iteration=model.best_iteration)

test_df = pd.DataFrame({'Id':Ids,'winPlacePerc':predictions})

print(test_df)

test_df.to_csv(r'submission.csv',index=False)
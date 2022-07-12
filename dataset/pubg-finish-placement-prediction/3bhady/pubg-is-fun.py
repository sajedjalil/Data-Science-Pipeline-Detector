# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.models import load_model

def split_train_val(data, fraction):
    matchIds = data['matchId'].unique().reshape([-1])
    train_size = int(len(matchIds)*fraction)
    
    random_idx = np.random.RandomState(seed=2).permutation(len(matchIds))
    train_matchIds = matchIds[random_idx[:train_size]]
    val_matchIds = matchIds[random_idx[train_size:]]
    
    data_train = data.loc[data['matchId'].isin(train_matchIds)]
    data_val = data.loc[data['matchId'].isin(val_matchIds)]
    return data_train, data_val




train =  pd.read_csv('../input/train_V2.csv')
data = train.copy()
data = data.sort_values("matchId")
data = data.reset_index(drop=True)
data = data[0:]

data[data == np.Inf] = np.NaN
data[data == np.NINF] = np.NaN
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


features = ["matchId", "winPlacePerc", "longestKill", "distancePerc", "killPerc","boostsPerc", "damageDealtPerc", "killPlacePerc", "killStreaksPerc", "swimDistancePerc", "rideDistancePerc", "vehicleDestroysPerc", "killPoints", "killPerc_walkDistance", "roadKillsPerc"]

data = data[features]

data[data == np.Inf] = np.NaN
data[data == np.NINF] = np.NaN
data.fillna(0, inplace=True)

train_set, val_set = split_train_val(data, 0.9)

y_train = train_set['winPlacePerc'].as_matrix()
y_val = val_set['winPlacePerc'].as_matrix()


features.remove("matchId")
features.remove("winPlacePerc")


train_set =train_set[features].as_matrix()
val_set =  val_set[features].as_matrix()

model = Sequential()
model.add(Dense(16, input_dim=13, activation='relu'))


model.add(Dense(1, activation='sigmoid'))


model.compile(loss='mse', optimizer='adam')

model.fit(train_set, y_train,
          batch_size=1024,
          epochs=30,
          verbose=1,
          validation_data=(val_set, y_val))
model.summary()
model.save("model.h5")

loaded_model = load_model("model.h5")
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

predictions = model.predict(test_set)
# print(predictions[0:10])
# [[0.5], [0.6], [0.8]]
test_df = pd.DataFrame({'Id':Ids,'winPlacePerc':predictions[:,0]})

print(test_df)

test_df.to_csv(r'submission.csv', index=False)
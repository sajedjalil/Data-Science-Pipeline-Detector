# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from time import time
import gc
gc.enable()

train_data = pd.read_csv('../input/train_V2.csv')
test_data = pd.read_csv('../input/test_V2.csv')

train_data.drop(['Id', 'groupId', 'matchId'], axis=1, inplace=True)
train_data.head()

train_data.columns

train_data[train_data['kills'] > 0].plot.scatter(x='winPlacePerc', y='kills', figsize=(12, 6))

l = ['winPlacePerc', 'boosts', 'damageDealt', 'heals', 'kills', 'rideDistance', 'roadKills', 'walkDistance', 'weaponsAcquired']
figure, ax = plt.subplots(figsize=(12,8))
f = train_data.loc[:, l].corr()
g = sns.heatmap(f, annot=True, ax=ax)
g.set_yticklabels(labels=l[::-1], rotation=0)
g.set_xticklabels(labels=l[:], rotation=90)

fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
train_data[train_data['walkDistance'] > 0].plot.scatter(x='winPlacePerc', y='walkDistance', ax=axarr[0][0])
train_data[train_data['boosts'] > 0].plot.scatter(x='winPlacePerc', y='boosts', ax=axarr[0][1])
train_data[train_data['weaponsAcquired'] > 0].plot.scatter(x='winPlacePerc', y='weaponsAcquired', ax=axarr[1][0])
train_data[train_data['damageDealt'] > 0].plot.scatter(x='winPlacePerc', y='damageDealt', ax=axarr[1][1])
plt.subplots_adjust(hspace=.3)
sns.despine()

train_data.columns

train_data[train_data['winPlacePerc'].isnull()]

train_data.drop(2744604, inplace=True)

figure1, axarr1 = plt.subplots(1, 3, figsize=(14, 6))
train_data['swimDistance'].value_counts().sort_index()[1:20].plot.hist(ax=axarr1[0])
train_data['rideDistance'].value_counts().sort_index()[1:20].plot.hist(ax=axarr1[1])
train_data['walkDistance'].value_counts().sort_index()[1:20].plot.hist(ax=axarr1[2])
axarr1[0].set_title('Swim dist')
axarr1[1].set_title('Ride dist')
axarr1[2].set_title('Walk dist')
plt.subplots_adjust(hspace=.3)
sns.despine()

train_data['totalDistance'] = train_data['swimDistance'] + train_data['rideDistance'] + train_data['walkDistance']
train_data.drop(['swimDistance', 'rideDistance', 'walkDistance'], axis=1, inplace=True)


train_data['matchType'].value_counts().index

# Creating cat codes of match type
train_data['matchType'] = train_data['matchType'].astype('category')
train_data['matchType'] = train_data['matchType'].cat.codes

# Combining boosts and heals as health
train_data['health'] = train_data['boosts'] + train_data['heals']
train_data.drop(['boosts', 'heals'], axis=1, inplace=True)

train_data.head()

figure2, axarr2 = plt.subplots(1, 3, figsize=(14, 6))
train_data['headshotKills'].value_counts().sort_index().head(10)[1:].plot.bar(ax=axarr2[0])
train_data['roadKills'].value_counts().sort_index().head(10)[1:].plot.bar(ax=axarr2[1])
train_data['teamKills'].value_counts().sort_index().head(10)[1:].plot.bar(ax=axarr2[2])
axarr2[0].set_title('Headshot kills')
axarr2[1].set_title('Road kills')
axarr2[2].set_title('Team kills')
plt.subplots_adjust(hspace=.3)
sns.despine()

train_data['kills'] += train_data['headshotKills'] + train_data['roadKills'] + train_data['teamKills']
train_data.drop(['headshotKills', 'roadKills', 'teamKills'], axis=1, inplace=True)

train_data.head()

test_data.drop(['Id', 'groupId', 'matchId'], axis=1, inplace=True)

test_data['totalDistance'] = test_data['swimDistance'] + test_data['rideDistance'] + test_data['walkDistance']
test_data.drop(['swimDistance', 'rideDistance', 'walkDistance'], axis=1, inplace=True)

test_data['matchType'] = test_data['matchType'].astype('category')
test_data['matchType'] = test_data['matchType'].cat.codes

test_data['health'] = test_data['boosts'] + test_data['heals']
test_data.drop(['boosts', 'heals'], axis=1, inplace=True)

test_data['kills'] += test_data['headshotKills'] + test_data['roadKills'] + test_data['teamKills']
test_data.drop(['headshotKills', 'roadKills', 'teamKills'], axis=1, inplace=True)

train_data.columns

# Mean
fill_NaN = Imputer(missing_values=np.nan, strategy='mean', axis=1)
train_data_mu = pd.DataFrame(fill_NaN.fit_transform(train_data))
train_data_mu.columns = train_data.columns
train_data_mu.index = train_data.index
train_data_mu.head()

y = train_data_mu['winPlacePerc']
X = train_data_mu.drop(['winPlacePerc'], axis=1)
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1511)

gc.collect()

clf = RandomForestRegressor(n_estimators=50, min_samples_leaf=3, min_samples_split=3, max_features=0.5 ,n_jobs=-1)
t0 = time()
clf.fit(train_X, train_y)
print('Training time', round(time() - t0, 3), 's')
pred = clf.predict(val_X)
print('MAE validation', mean_absolute_error(val_y, pred))

# Mean
fill_NaN_test = Imputer(missing_values=np.nan, strategy='mean', axis=1)
test_data_mu = pd.DataFrame(fill_NaN_test.fit_transform(test_data))
test_data_mu.columns = test_data.columns
test_data_mu.index = test_data.index
test_data_mu.head()

test_data = pd.read_csv('../input/test_V2.csv')
pred1 = clf.predict(test_data_mu)
test_data['winPlacePerc'] = pred1
submission = test_data[['Id', 'winPlacePerc']]
submission.to_csv('output.csv', index=False)

# Any results you write to the current directory are saved as output.
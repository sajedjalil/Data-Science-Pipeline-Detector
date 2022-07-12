import kagglegym
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
import xgboost as xgb

target = 'y'

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train
mean_values = train.mean(axis=0)
train.fillna(mean_values, inplace=True)

# Observed with histograns:
low_y_cut = -0.086093
high_y_cut = 0.093497

y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

train['y'] = train['y'] * 5 + 0.5
print(train['y'].describe())

print(np.sqrt(np.mean((train['y'].values - train['y'].mean())**2)))

print('Splitting data')

df_train = train[:600000]
df_valid = train[600000:]
del train

y_train = df_train['y'].values
y_valid = df_valid['y'].values

x_train = df_train.drop(['id', 'y', 'timestamp'], axis=1)
x_valid = df_valid.drop(['id', 'y', 'timestamp'], axis=1)
del df_train, df_valid

print(np.sqrt(np.mean((y_train - y_train.mean())**2)))
print(np.sqrt(np.mean((y_valid)**2)))

print('Creating datamatrices')

print(y_train)
print(y_valid)

d_train = xgb.DMatrix(x_train, label=y_train)
del x_train
d_valid = xgb.DMatrix(x_valid, label=y_valid)
del x_valid

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

params = {}
params['eta'] = 0.1
params['booster'] = 'gbtree'
params['base_score'] = 0.5
params['max_depth'] = 6
params['silent'] = 1

clf = xgb.train(params, d_train, 50, watchlist, early_stopping_rounds=50)

while True:
    print('Running for test.')
    test_x = observation.features.fillna(mean_values)
    #test_x = np.array(observation.features[col].values).reshape(-1,1)
    observation.target.y = clf.predict(xgb.DMatrix(test_x)).clip(low_y_cut, high_y_cut)
    #observation.target.fillna(0, inplace=True)
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    observation, reward, done, info = env.step(target)
    if done:
        break
    
info



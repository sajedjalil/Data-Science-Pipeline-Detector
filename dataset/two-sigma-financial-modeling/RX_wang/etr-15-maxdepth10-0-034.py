import numpy as np
import pandas as pd
import kagglegym

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

train = observation.train
# y_within_cut = (train.y < 0.093) & (train.y > -0.086)
# train = train[y_within_cut]
train = train.sort_values(['id', 'timestamp'])
cols = [col for col in train.columns if col != 'y']
for col in cols:
    train[col + '_1'] = train[col] -  train[col].groupby(train.id).shift(1)
cols_use = ['technical_20', 'technical_20_1', 'technical_30', 'technical_19',
       'technical_43', 'technical_36', 'technical_0_1', 'technical_35_1',
       'fundamental_41', 'technical_35', 'fundamental_3', 'fundamental_33',
       'technical_6', 'technical_34_1', 'technical_11', 'timestamp', 'id']

y = train.y
train = train[cols_use]
X = train.drop(['timestamp', 'id'], axis=1)

mean_X = X.mean()
X = X.fillna(mean_X)
cols = ['technical_20', 'technical_0', 'technical_35', 'technical_34', 'id']
test = train.ix[train.timestamp == 905, cols]
test.columns = [col + '_' for col in cols]

print('Begin fitting!')
from sklearn.ensemble import ExtraTreesRegressor
etr = ExtraTreesRegressor(n_jobs=-1, max_depth=10, n_estimators=100, bootstrap=True)
etr.fit(X, y)

print('Begin Predict')
while True:
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    X_test = pd.merge(observation.features, test, left_on='id', right_on='id_', how='left')
    for col in cols:
        X_test[col + '_1'] = X_test[col] - X_test[col + '_']
    X = X_test[cols_use]
    
    X = X.fillna(mean_X)

    X = X.drop(['timestamp', 'id'], axis=1)
    
    test = X_test[cols]
    test.columns = [col + '_' for col in cols]
    
    target.y = etr.predict(X)

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break
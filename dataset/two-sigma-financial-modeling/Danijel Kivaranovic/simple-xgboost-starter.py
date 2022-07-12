import kagglegym
import numpy as np
import pandas as pd
import xgboost as xgb

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

## features
feat = ['fundamental_11', 'technical_19', 'technical_20', 'technical_30']

## training data
dtrain = xgb.DMatrix(observation.train[feat], label = observation.train['y'].values)

## parameters
param = {'booster':           'gbtree',
         'objective':         'reg:linear',
         'learning_rate':     0.025,
         'max_depth':         2,
         'subsample':         0.5,
         'colsample_bytree':  0.7,
         'colsample_bylevel': 0.7,
         'silent':            1
}

## train model
print('train model...')
bst = xgb.train(params = param,
                dtrain = dtrain,
                num_boost_round = 500)

print('predicting...')
while True:
    timestamp = observation.features['timestamp'][0]
    target = observation.target
    dtest = xgb.DMatrix(observation.features[feat])
    pred = bst.predict(dtest)
    observation.target.y = pred
    
    if timestamp % 100 == 0:
        print('Timestamp #{}'.format(timestamp))
    
    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print('Public score: {}'.format(info['public_score']))
        break
        
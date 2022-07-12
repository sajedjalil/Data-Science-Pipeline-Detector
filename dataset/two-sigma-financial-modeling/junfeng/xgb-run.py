import kagglegym
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
import xgboost as xgb

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train
mean_values = train.mean(axis=0)
train.fillna(mean_values, inplace=True)

cols_to_use = ['technical_30', 'technical_20',
  'fundamental_11',
  'technical_19']
# cols_to_use = [col for col in train.columns if col not in ['id','timestamp','y']]

print(cols_to_use)

dtrain = xgb.DMatrix(train[cols_to_use].values, train.y.values)

params = {'booster':'gbtree',
          'max_depth':9,
          'eta':0.03,
          'silent':1,
          'objective':'reg:linear',
          'subsample':0.7,
          
}
params['eval_metric'] = 'rmse'
num_round = 300
evallist  = [(dtrain,'train')]

bst = xgb.train(params, dtrain, num_round, evallist)

print('Predicting ...')
while True:
    observation.features.fillna(mean_values, inplace=True)
    test_x = observation.features[cols_to_use]
    dtest = xgb.DMatrix(test_x.values)
    ypred = bst.predict(dtest)
    observation.target.y = ypred
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        
    observation, reward, done, info = env.step(target)
    if done:
        break
print(info)

# xgb.plot_importance(bst)
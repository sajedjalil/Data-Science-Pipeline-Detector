import kagglegym
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))

# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

train = observation.train
median_values = train.median(axis=0)
train.fillna(median_values, inplace=True)

#feat = [name for name in observation.train.columns if name not in ['id', 'timestamp', 'y']]
feat = ['technical_20', 'fundamental_53', 'technical_30', 'technical_27', 'derived_0',\
        'fundamental_42', 'fundamental_48', 'technical_21', 'technical_24', 'fundamental_11']

low_y_cut = -0.086093
high_y_cut = 0.093497

y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)


dtrain = xgb.DMatrix(train.loc[y_is_within_cut, feat].values, \
                     label = train.loc[y_is_within_cut, 'y'].values)

params = {}
params['booster']  = 'gbtree'
params['objective'] = 'reg:linear'
params['max_depth'] = 6
params['subsample'] = 0.8
params['colsample_bytree'] = 0.8
params['silent'] = 1
params['eval_metric'] = 'rmse'
num_round = 50
eval_list  = [(dtrain,'train')]

print('training xgb model...')
bst = xgb.train(params, dtrain, num_round, eval_list)

print('training ridge regression...')
lr = Ridge()
lr.fit(train.loc[y_is_within_cut, feat].values, \
       train.loc[y_is_within_cut, 'y'].values)

while True:
    observation.features.fillna(median_values, inplace=True)
    dtest = xgb.DMatrix(observation.features[feat].values)
    xgb_pred = bst.predict(dtest).clip(low_y_cut, high_y_cut)
    lr_pred = lr.predict(observation.features[feat].values).clip(low_y_cut, high_y_cut)
    observation.target.y = 0.2*xgb_pred+0.8*lr_pred

    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break

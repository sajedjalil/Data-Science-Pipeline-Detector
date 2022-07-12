import kagglegym
import numpy as np
import pandas as pd
import xgboost as xgb

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()
ee = observation.train
features = ee.ix[:,[59,87]].fillna(0)
low_y_cut = -0.026093
high_y_cut = 0.023497

y_is_above_cut = (ee.y > high_y_cut)
y_is_below_cut = (ee.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

T_train_xgb = xgb.DMatrix(features.loc[y_is_within_cut,:],ee.y.loc[y_is_within_cut])
params = {"objective": "reg:linear", "booster":"gblinear", "alpha":0.0}
gbm = xgb.train(dtrain=T_train_xgb,params=params)


while True:
    target = observation.target
    target.y = gbm.predict(xgb.DMatrix(observation.features.ix[:,[59,87]].fillna(0))).clip(low_y_cut, high_y_cut)
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break
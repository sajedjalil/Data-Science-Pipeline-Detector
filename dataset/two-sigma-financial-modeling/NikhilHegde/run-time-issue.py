import kagglegym
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

env = kagglegym.make()
observation = env.reset()

train = observation.train
train['timestamp'] = pd.to_datetime(train.timestamp)
train = train.sort_values(by = 'timestamp').fillna(method = 'ffill')
train = train.fillna(method = 'bfill')
d_mean= train.median(axis=0)
train = train.fillna(d_mean)
train = train.sort_values(by =['timestamp','id'],ascending = [True,True]).reset_index(drop = True)

#feat = [name for name in observation.train.columns if name not in ['id', 'timestamp', 'y']]
feat = ['technical_20', 'fundamental_53', 'technical_30', 'technical_27', 'derived_0',\
        'fundamental_42', 'fundamental_48', 'technical_21', 'technical_24', 'fundamental_11','technical_40']
cols_to_use = ['technical_30', 'technical_20', 'fundamental_11']

low_y_cut = -0.075
high_y_cut = 0.075
y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
       
rfr = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=17, verbose=0)
model1 = rfr.fit(train.loc[y_is_within_cut, feat].values, train.loc[y_is_within_cut, 'y'].values)

model2 = LinearRegression(n_jobs=-1)
model2 = model2.fit(np.array(train[feat].loc[y_is_within_cut, 'technical_20'].values).reshape(-1,1), train.loc[y_is_within_cut, 'y'])


while True:
    test = observation.features
    test['timestamp'] = pd.to_datetime(test.timestamp)
    test.fillna(method = 'ffill', inplace=True)
    test.fillna(method = 'bfill', inplace = True)
    d_mean= test.mean(axis=0)
    test.fillna(d_mean, inplace = True)
    test = test.sort_values(by =['timestamp','id'],ascending = [True,True]).reset_index(drop = True)
    rfr_pred = model1.predict(test[feat]).clip(low_y_cut, high_y_cut)
    lr_pred = model2.predict(np.array(test['technical_20'].values).reshape(-1,1))
    observation.target.y = 0.05*rfr_pred+0.95*lr_pred

    target = observation.target
    timestamp = observation.features["timestamp"][0]
    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break

import kagglegym
import numpy as np
import pandas as pd
import xgboost as xgb

env = kagglegym.make()
obs = env.reset()
train = pd.read_hdf('../input/train.h5')

train_x = train.drop(['id','y','timestamp'],axis = 1)
train_y = train['y']
train_x.fillna(train.mean(),inplace = True)
train_x.fillna(0,inplace=True)

xgbmat = xgb.DMatrix(train_x, train_y)

params = {'eta': 1.0,'subsample': 0.5,'min_child_weight':4,
          'colsample_bytree':0.5,'objective': 'reg:linear', 'max_depth':8,'alpha':1,
          } 
          
num_rounds = 1000
bst = xgb.train(params, xgbmat, num_boost_round = num_rounds)

while True:
    test_x = obs.features
    test_x = test_x.drop(['id','timestamp'],axis = 1)
    test_xgb = xgb.DMatrix(test_x)
    pred = obs.target
    pred['y'] = bst.predict(test_xgb)
    obs,reward,done,info = env.step(pred)
    if done:
            print(info["public_score"])
            break;
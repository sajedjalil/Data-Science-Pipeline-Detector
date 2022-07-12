import kagglegym
import time
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
import math

def time_elapsed(t0):
    return (time.time()-t0)/60

def _reward(y_true, y_fit):
    R2 = 1 - np.sum((y_true - y_fit)**2) / np.sum((y_true - np.mean(y_true))**2)
    R = np.sign(R2) * math.sqrt(abs(R2))
    return(R)

env = kagglegym.make()
o = env.reset()
excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
col = [c for c in o.train.columns if c not in excl]
col = ['technical_20','technical_40']
train = pd.read_hdf('../input/train.h5')
train = train[col]
#d_mean = train.mean(axis=0)
d_mean= train.median(axis=0)
#d_mode = train.mode(axis=0) #takes long
#d_min = train.min(axis=0)
#d_max = train.max(axis=0)

train = o.train[col]
n = train.isnull().sum(axis=1)
for c in train.columns:
    train[c + '_nan_'] = pd.isnull(train[c])
    d_mean[c + '_nan_'] = 0
train = train.fillna(d_mean)
train['znull'] = n
n = []

'''
# Randomized Trees Regressor Model
t0 = time.time()
rfr = ExtraTreesRegressor(n_estimators=80, max_depth=4, max_features='auto', n_jobs=-1, random_state=17, verbose=0)
print("RTR Model Training started..")
rfrmodel = rfr.fit(train, o.train['y'])
print("RTR Model Training completed!")
print(time_elapsed(t0), "minutes elapsed!")

#https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189
'''
low_y_cut = -0.075
high_y_cut = 0.075
y_is_above_cut = (o.train.y > high_y_cut)
y_is_below_cut = (o.train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
'''
print("Linear Model Training started..")
t0 = time.time()
model2 = LinearRegression(n_jobs=-1)
model2.fit(np.array(o.train[col].fillna(d_mean).loc[y_is_within_cut, 'technical_20'].values).reshape(-1,1), o.train.loc[y_is_within_cut, 'y'])
print("Linear Model Training completed!")
print(time_elapsed(t0), "minutes elapsed!")
'''

# Putting XGB Model
print("Started XGB DMatrix Preparation..")
t0 = time.time()
dtrain = xgb.DMatrix(train.values, label=o.train['y'].values)
print("DMatrix created!")
print(time_elapsed(t0), "minutes elapsed!")
print("Rows %d"%dtrain.num_row())
print("Cols %d"%dtrain.num_col())

del train
gc.collect()

param = {'max_depth':4, 'eta':0.4, 'silent':1, 'objective':'reg:linear' , 'eval_metric':'mae'}
#param['eval_metric'] = 'auc'
param['subsample'] = .5
param['colsample_bytree']= 1
param['min_child_weight'] = 10
param['booster'] = "gblinear"
param['gamma'] = 100
param['seed'] = 12
watchlist = [(dtrain,'train')]
#early_stopping_rounds = 10
t0 = time.time()
print("XGB Model Training started..")
num_train_round = 40
xgbModel = xgb.train(param, 
                     dtrain, 
                     num_boost_round=num_train_round,
                     evals=watchlist, 
                     #feval=feval_matthews, 
                     #maximize=True, 
                     verbose_eval=2)
print("XGB Model Training completed!")
print(time_elapsed(t0), "minutes elapsed!")
del dtrain
gc.collect()

train = []

#https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659
ymean_dict = dict(o.train.groupby(["id"])["y"].median())
def get_weighted_y(series):
    id, y = series["id"], series["y"]
    return 0.95 * y + 0.05 * ymean_dict[id] if id in ymean_dict else y

i = 0; reward_=[]
while True:
    test = o.features[col]
    n = test.isnull().sum(axis=1)
    for c in test.columns:
        test[c + '_nan_'] = pd.isnull(test[c])
    test = test.fillna(d_mean)
    test['znull'] = n
    pred = o.target
    dtest = xgb.DMatrix(test.values)   
    test2 = np.array(o.features[col].fillna(d_mean)['technical_20'].values).reshape(-1,1)
    #pred['y'] = (0.65 * model1.predict(test).clip(low_y_cut, high_y_cut) + 0.35 * model2.predict(test2).clip(low_y_cut, high_y_cut))
    #pred['y'] = (0.2 * xgbModel.predict(dtest).clip(low_y_cut, high_y_cut) + 0.5 * rfrmodel.predict(test).clip(low_y_cut, high_y_cut) + 0.3 * model2.predict(test2).clip(low_y_cut, high_y_cut))
    pred['y'] = xgbModel.predict(dtest).clip(low_y_cut, high_y_cut)
    # print(pred['y'])
    pred['y'] = pred.apply(get_weighted_y, axis = 1)
    del dtest
    gc.collect()
    o, reward, done, info = env.step(pred[['id','y']])
    reward_.append(reward)
    if i % 50 == 0:
        # print(pred['y'])
        print(i, reward, np.mean(np.array(reward_)))
    
    i += 1
    if done:
        print("el fin ...", info["public_score"])
        break
  
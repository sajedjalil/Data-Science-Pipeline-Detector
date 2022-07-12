import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
import math

def _reward(y_true, y_fit):
    R2 = 1 - np.sum((y_true - y_fit)**2) / np.sum((y_true - np.mean(y_true))**2)
    R = np.sign(R2) * math.sqrt(abs(R2))
    return(R)

env = kagglegym.make()
o = env.reset()
excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
col = [c for c in o.train.columns if c not in excl]

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

rfr = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=308537, verbose=0)
model1 = rfr.fit(train, o.train['y'])

#https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189
low_y_cut = -0.075
high_y_cut = 0.075
y_is_above_cut = (o.train.y > high_y_cut)
y_is_below_cut = (o.train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
model2 = LinearRegression(n_jobs=-1)
model2.fit(np.array(o.train[col].fillna(d_mean).loc[y_is_within_cut, 'technical_20'].values).reshape(-1,1), o.train.loc[y_is_within_cut, 'y'])
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
    test2 = np.array(o.features[col].fillna(d_mean)['technical_20'].values).reshape(-1,1)
    pred['y'] = (model1.predict(test).clip(low_y_cut, high_y_cut) + model2.predict(test2).clip(low_y_cut, high_y_cut))/2
    pred['y'] = pred.apply(get_weighted_y, axis = 1)
    o, reward, done, info = env.step(pred[['id','y']])
    reward_.append(reward)
    if i % 100 == 0:
        print(reward, np.mean(np.array(reward_)))
    i += 1
    if done:
        print("el fin ...", info["public_score"])
        break
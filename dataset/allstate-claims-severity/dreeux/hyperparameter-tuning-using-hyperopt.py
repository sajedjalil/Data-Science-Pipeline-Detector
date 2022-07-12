
# http://www.dataiku.com/blog/2015/08/24/xgboost_and_dss.html
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import matplotlib.pylab as plt
import pandas as pd, numpy as np
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from hyperopt import hp, tpe, STATUS_OK, Trials
#from hyperopt.fmin import fmin

target = 'loss'
ID = 'id'
DATA_DIR = "../input"
TRAIN_FILE = "{0}/train.csv".format(DATA_DIR)

data = pd.read_csv(TRAIN_FILE)
data.drop([ID], axis=1, inplace=True)

train = data.sample(frac=0.8, random_state=1)
valid = data.drop(train.index)

features = train.columns

cats = [feat for feat in features if 'cat' in feat]
for feat in cats:
    train[feat] = pd.factorize(train[feat], sort=True)[0]
    valid[feat] = pd.factorize(valid[feat], sort=True)[0]


y_train = (train[target])
y_valid = (valid[target])

#y_train = np.log(train[target].ravel())
#y_valid = np.log(valid[target].ravel())

del train[target]
del valid[target]

col_train = train.columns

def xgb_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))

def objective(space):

    clf = xgb.XGBRegressor(n_estimators = space['n_estimators'],
                           max_depth = space['max_depth'],
                           min_child_weight = space['min_child_weight'],
                           subsample = space['subsample'],
                           learning_rate = space['learning_rate'],
                           gamma = space['gamma'],
                           colsample_bytree = space['colsample_bytree'],
                           objective='reg:linear'
                           )

    eval_set  = [( train, y_train), ( valid, y_valid)]

    clf.fit(train[col_train],
            y_train,
            eval_set=eval_set,
            eval_metric = 'mae')

    pred = clf.predict(valid)
#   mae = mean_absolute_error(np.exp(y_valid), np.exp(pred))
    mae = mean_absolute_error((y_valid), (pred))

#    print "SCORE:", mae
    return{'loss':mae, 'status': STATUS_OK }


space ={
        'max_depth': hp.choice('max_depth', np.arange(10, 30, dtype=int)),
        'min_child_weight': hp.quniform ('min_child', 1, 20, 1),
        'subsample': hp.uniform ('subsample', 0.8, 1),
        'n_estimators' : hp.choice('n_estimators', np.arange(1000, 10000, 100, dtype=int)),
        'learning_rate' : hp.quniform('learning_rate', 0.025, 0.5, 0.025),
        'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05)
    }


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=3, # change
            trials=trials)

print(best)

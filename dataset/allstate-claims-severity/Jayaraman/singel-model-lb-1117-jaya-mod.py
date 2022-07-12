# -*- coding: utf-8 -*-
"""
Based on Faron' script 
https://www.kaggle.com/mmueller/allstate-claims-severity/stacking-starter/run/390867
"""
## loading packages 
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
import gc



### setup
ID = 'id'
TARGET = 'loss'
NFOLDS = 5
SEED = 1
NROWS = None
DATA_DIR = "../input"
OUT_DIR ="../output"


### reading data 
TRAIN_FILE = "{0}/train.csv".format(DATA_DIR)
TEST_FILE = "{0}/test.csv".format(DATA_DIR)
SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)

train = pd.read_csv(TRAIN_FILE, nrows=NROWS)
test = pd.read_csv(TEST_FILE, nrows=NROWS)



ntrain = train.shape[0]
ntest = test.shape[0]
y_train = np.log(train[TARGET]).ravel()
id_train= train[ID]
id_test= test[ID]

train_test = pd.concat((train, test)).reset_index(drop=True)

### remeber the order
train_test[ID]=pd.Categorical(train_test[ID], train_test[ID].values.tolist())

### factorize
cats = [feat for feat in train.columns if 'cat' in feat]
for cat in cats:
    sorting_list=np.unique(sorted(train_test[cat],key=lambda x:(str.lower(x),x)))
    train_test[cat]=pd.Categorical(train_test[cat], sorting_list)
    train_test=train_test.sort_values(cat)
    train_test[cat] = pd.factorize(train_test[cat], sort=True)[0]

### reorder 
train_test=train_test.sort_values(ID)
gc.collect()


#### preprocessing
train_test["cont1"] = np.sqrt(preprocessing.minmax_scale(train_test["cont1"]))
train_test["cont4"] = np.sqrt(preprocessing.minmax_scale(train_test["cont4"]))
train_test["cont5"] = np.sqrt(preprocessing.minmax_scale(train_test["cont5"]))
train_test["cont8"] = np.sqrt(preprocessing.minmax_scale(train_test["cont8"]))
train_test["cont10"] = np.sqrt(preprocessing.minmax_scale(train_test["cont10"]))
train_test["cont11"] = np.sqrt(preprocessing.minmax_scale(train_test["cont11"]))
train_test["cont12"] = np.sqrt(preprocessing.minmax_scale(train_test["cont12"]))

train_test["cont6"] = np.log(preprocessing.minmax_scale(train_test["cont6"])+0000.1)
train_test["cont7"] = np.log(preprocessing.minmax_scale(train_test["cont7"])+0000.1)
train_test["cont9"] = np.log(preprocessing.minmax_scale(train_test["cont9"])+0000.1)
train_test["cont13"] = np.log(preprocessing.minmax_scale(train_test["cont13"])+0000.1)
train_test["cont14"]=(np.maximum(train_test["cont14"]-0.179722,0)/0.665122)**0.25


### define x_train, x_test
train_test.drop([ID, TARGET], axis=1, inplace=True)
x_train = np.array(train_test.iloc[:ntrain,:])
x_test = np.array(train_test.iloc[ntrain:,:])
print("{},{}".format(x_train.shape, x_test.shape))



### setup training functions

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)



class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


def get_oof(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
    
#### xgb
xgb_params = {
    'seed': 1,
    'colsample_bytree': 0.3085,
    'subsample': 0.9930,
    'eta': 0.1,
    'lambda':.5,
    'gamma': 0.49,
    'booster' :  'gbtree',    
    'objective': 'reg:linear',
    'max_depth': 10,
    'min_child_weight': 4.28,
    'eval_metric': 'mae'
}


# LB 1117.65579
#xgb_params['nrounds']=xgb2_rounds
xgb_params['nrounds']=493
print(xgb_params)

xgb_model = XgbWrapper(seed=SEED, params=xgb_params)
xgb_oof_train, xgb_oof_test = get_oof(xgb_model)
print("XGB-CV: {}".format(mean_absolute_error(y_train, xgb_oof_train)))

xgb_test = pd.DataFrame(np.exp(xgb_oof_test), columns=[TARGET])
xgb_test[ID] = id_test
xgb_test.to_csv('xgb_test.csv', index=0)




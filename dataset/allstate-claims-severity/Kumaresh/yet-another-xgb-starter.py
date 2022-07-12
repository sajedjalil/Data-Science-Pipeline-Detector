# -*- coding: utf-8 -*-
"""
@author: Faron
"""
import pandas as pd
import numpy as np
import xgboost as xgb

ID = 'id'
TARGET = 'loss'
SEED = 0
DATA_DIR = "../input"

TRAIN_FILE = "{0}/train.csv".format(DATA_DIR)
TEST_FILE = "{0}/test.csv".format(DATA_DIR)
SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)


train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

y_train1 = np.log(train[TARGET].ravel())
y_train2 = (train[TARGET].ravel())

train.drop([ID, TARGET], axis=1, inplace=True)
test.drop([ID], axis=1, inplace=True)

print("{},{}".format(train.shape, test.shape))

ntrain = train.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True)

features = train.columns

cats = [feat for feat in features if 'cat' in feat]
for feat in cats:
    train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]

print(train_test.head())

x_train = np.array(train_test.iloc[:ntrain,:])
x_test = np.array(train_test.iloc[ntrain:,:])

print("{},{}".format(train.shape, test.shape))

dtrain1 = xgb.DMatrix(x_train, label=y_train1)
dtrain2 = xgb.DMatrix(x_train, label=y_train2)

dtest = xgb.DMatrix(x_test)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.05,
    'objective': 'reg:linear',
    'max_depth': 6,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'mae',
}

res = xgb.cv(xgb_params, dtrain1, num_boost_round=800, nfold=4, seed=SEED, stratified=False,
             early_stopping_rounds=5, verbose_eval=10, show_stdv=True)

xgb_params2 = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.05,
    'objective': 'reg:gamma',
    'max_depth': 6,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'mae',
}

res2 = xgb.cv(xgb_params2, dtrain2, num_boost_round=500, nfold=4, seed=SEED, stratified=False,
             early_stopping_rounds=5, verbose_eval=10, show_stdv=True)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]
best_nrounds2 = res2.shape[0] - 1
cv_mean2 = res2.iloc[-1, 0]
cv_std2 = res2.iloc[-1, 1]
print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))
print('CV-Mean: {0}+{1}'.format(cv_mean2, cv_std2))

gbdt = xgb.train(xgb_params, dtrain1, best_nrounds)
gbdt2 = xgb.train(xgb_params2, dtrain2, best_nrounds)


submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = 0.5*np.exp(gbdt.predict(dtest))+0.5*gbdt2.predict(dtest)

submission.to_csv('xgb_starter.sub.csv', index=None)
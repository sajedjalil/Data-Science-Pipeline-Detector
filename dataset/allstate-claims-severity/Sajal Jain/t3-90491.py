# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import logging

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression


from heamy.dataset import Dataset
from heamy.estimator import Regressor
from heamy.pipeline import ModelsPipeline

logging.basicConfig(level=logging.DEBUG)
ID = 'id'
TARGET = 'loss'

DATA_DIR = "../input"
SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)


def first_dataset():
    train = pd.read_csv("{0}/train.csv".format(DATA_DIR))
    test = pd.read_csv("{0}/test.csv".format(DATA_DIR))

    y_train = train[TARGET].ravel()

    train.drop([ID, TARGET], axis=1, inplace=True)
    test.drop([ID], axis=1, inplace=True)

    train_test = pd.concat((train, test)).reset_index(drop=True)

    ntrain = train.shape[0]

    features = train.columns
    cats = [feat for feat in features if 'cat' in feat]
    for feat in cats:
        train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]
    x_train = np.array(train_test.iloc[:ntrain, :])
    x_test = np.array(train_test.iloc[ntrain:, :])

    return {'X_train': x_train, 'X_test': x_test, 'y_train': y_train}


def xgb_first(X_train, y_train, X_test, y_test=None):
    params = {
        'seed': 1111,
        'colsample_bytree': 0.7,
        'silent': 1,
        'subsample': 0.8,
        'learning_rate': 0.01,
        'objective': 'reg:linear',
        'max_depth': 8,
        'num_estimators': 350,
        'min_child_weight': 1,
    }

    X_train = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, X_train, params['num_estimators'], )
    return model.predict(xgb.DMatrix(X_test))


def xgb_stack(X_train, y_train, X_test, y_test=None):
    params = {
        'seed': 3333,
        'colsample_bytree': 0.6,
        'silent': 1,
        'subsample': 0.85,
        'learning_rate': 0.005,
        'objective': 'reg:linear',
        'max_depth': 10,
        'num_estimators': 550,
        'gamma': 0.005,
    }
    X_train = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
    model = xgb.train(params, X_train, params['num_estimators'], )
    return model.predict(xgb.DMatrix(X_test, missing=np.nan))


et_params = {'n_estimators': 100, 'max_features': 0.5,
             'max_depth': 18, 'min_samples_leaf': 4,
             'n_jobs': -1}
rf_params = {'n_estimators': 125, 'max_features': 0.2,
             'max_depth': 25, 'min_samples_leaf': 4,
             'n_jobs': -1}

ds = Dataset(preprocessor=first_dataset, use_cache=False)
pipeline = ModelsPipeline(
    Regressor(estimator=xgb_first, dataset=ds, use_cache=False),
    Regressor(estimator=ExtraTreesRegressor, dataset=ds, use_cache=False,
              parameters=et_params),
    Regressor(estimator=RandomForestRegressor, dataset=ds, use_cache=False,
              parameters=rf_params),
    Regressor(estimator=LinearRegression, dataset=ds, use_cache=False),

)
# 4 folds
stack_ds = pipeline.stack(k=4, seed=111, add_diff=False, full_test=True)

# One model approach
# stacker = Regressor(dataset=stack_ds, estimator=xgb_stack, use_cache=False)
# Uncomment for valdation
# stacker.validate(k=2, scorer=mean_absolute_error)
# predictions = stacker.predict()

# Two models on the second layer
pipe2 = ModelsPipeline(
    Regressor(estimator=xgb_stack, dataset=stack_ds, use_cache=False),
    Regressor(estimator=ExtraTreesRegressor, dataset=stack_ds, use_cache=False,
              parameters={'n_estimators': 100, 'max_depth': 15, 'max_features': 3}),

)
# pipe2.weight([0.75, 0.25]).validate(k=2, scorer=mean_absolute_error)
# xgb*0.75+rf*0.25
predictions = pipe2.weight([0.75, 0.25]).execute()

submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = predictions
submission.to_csv('xgstacker_starter.sub.csv', index=None)
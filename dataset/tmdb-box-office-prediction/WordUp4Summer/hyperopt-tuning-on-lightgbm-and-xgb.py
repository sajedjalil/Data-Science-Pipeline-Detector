# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from sklearn.model_selection import train_test_split, KFold

# Libraries
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, KFold

import xgboost as xgb
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import json
import ast
import eli5
import shap
from catboost import CatBoostRegressor
from urllib.request import urlopen
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

from hyperopt import hp, tpe
from hyperopt.fmin import fmin

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
import lightgbm as lgbm

## reference kernel 1: my data and feature engineering was from there https://www.kaggle.com/artgor/eda-feature-engineering-and-model-interpretation
## reference kernel 2: https://www.kaggle.com/eikedehling/tune-and-compare-xgb-lightgbm-rf-with-hyperopt
#os.listdir('..')
#print(check_output(["ls", "../input"]).decode("utf8"))
## load data
X = pd.read_csv('../input/x-from-kernel/X_from_kernel.csv')
X_test = pd.read_csv('../input/x-test-from-kernel/X_test_from_kernel.csv')
train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

## prepossesing
y = np.log1p(train['revenue'])
X.set_index(X.columns[0], inplace=True)
X_test.set_index(X_test.columns[0], inplace=True)

#### tune XGB
def objective(params):
    params = {
        'max_depth': int(params['max_depth']),
        'gamma': "{:.3f}".format(params['gamma']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
    }
    
    reg = xgb.XGBRegressor(
        n_estimators=250,
        learning_rate=0.05,
        n_jobs=4,
        
        **params
    )
    
    score = cross_val_score(reg, X, y, scoring='neg_mean_squared_error', cv=KFold(n_splits=5)).mean()
    print("neg_mean_s_e {:.3f} params {}".format(score, params))
    return score
## search space
space = {
    'max_depth': hp.quniform('max_depth', 6, 9, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'gamma': hp.uniform('gamma', 0.0, 0.5),
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest, max_evals=10)  ## i used max_eval= 30
            
## this is the best xgb params from the hyperopt max_eval = 30 combined
best_xgb_params = {'colsample_bytree': 0.7981233908710783,
 'gamma': 0.36734395511936885,
 'max_depth': int(9.0)}
#best_xgb_params = best
 
#### Tune lighGBM
def objective(params):
    params = {
        'num_leaves': int(params['num_leaves']),
        'min_data_in_leaf': int(params['min_data_in_leaf']),
        'max_depth': int(params['max_depth']),
        'lambda_l1': '{:.3f}'.format(params['lambda_l1']),
    }
    
    reg = lgbm.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.01,
        **params
    )
    
    score = cross_val_score(reg, X, y, scoring='neg_mean_squared_error', cv=KFold(n_splits=5)).mean()
    print("neg_mean_s_e {:.3f} params {}".format(score, params))
    return score

space = {
    'num_leaves': hp.quniform('num_leaves', 10, 50, 5),
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 50, 5),
    'max_depth': hp.quniform('max_depth', 2, 10, 1),
    'lambda_l1': hp.uniform('lambda_l1', 0.01, 0.3)

}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10)

## best for lighGBM
best_lgb_params = {'lambda_l1': 0.29513409605971064,
 'max_depth': int(2.0),
 'min_data_in_leaf': int(40.0),
 'num_leaves': int(45)}
#best_lgb_params = best

#### model comparison
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)
def train_model(X, X_test, y, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None):

    #oof = np.zeros(X.shape[0])
    prediction = np.zeros(X_test.shape[0])
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.values[train_index], X.values[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        
        if model_type == 'lgb':
            model = lgbm.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',
                    verbose=1000, early_stopping_rounds=200)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test.values), ntree_limit=model.best_ntree_limit)

    
        
        scores.append(mean_squared_error(y_valid, y_pred_valid) ** 0.5)
        
        prediction += y_pred    
        
        
    prediction /= n_fold
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

for param, model in zip([best_lgb_params, best_xgb_params], ['lgb', 'xgb']):
    print('######### ', model, ' #########')
    train_model(X, X_test, y, params=param, model_type=model)
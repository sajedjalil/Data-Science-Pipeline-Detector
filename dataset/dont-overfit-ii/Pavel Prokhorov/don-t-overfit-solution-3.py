import numpy as np
import pandas as pd

from copy import deepcopy

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score, make_scorer, r2_score
from sklearn.feature_selection import RFECV

import warnings
warnings.simplefilter('ignore')

import os
print(os.listdir("../input"))




#
# Framework
#

class Estimator(object):
    
    def get_estimator(self):
        raise NotImplementedException
    
    def fit(self, x_train, y_train, x_valid, y_valid):
        raise NotImplementedException
    
    def predict(self, x):
        raise NotImplementedException


class ScikitLearnEstimator(Estimator):
    
    def __init__(self, estimator):
        self.estimator = estimator
    
    def get_estimator(self):
        return self.estimator
    
    def fit(self, x_train, y_train, x_valid, y_valid):
        self.estimator.fit(x_train, y_train)
    
    def predict(self, x):
        return self.estimator.predict(x)


class GridSearchCvEstimator(Estimator):
    
    def __init__(self, estimator, param_grid, scoring, cv):
        self.grid = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring, cv=cv, verbose=0, n_jobs=-1)
    
    def get_estimator(self):
        return self.grid.best_estimator_
    
    def fit(self, x_train, y_train, x_valid, y_valid):
        
        self.grid.fit(x_train, y_train)
        
        print('Best score:', self.grid.best_score_)
        print('Best parameters', self.grid.best_params_)
    
    def predict(self, x):
        return self.grid.best_estimator_.predict(x)


def fit_step(estimator, scoring, x_train, y_train, train_idx, valid_idx):
    
    # prepare train and validation data
    x_train_train = x_train[train_idx]
    y_train_train = y_train[train_idx]
    x_train_valid = x_train[valid_idx]
    y_train_valid = y_train[valid_idx]
    
    # fit estimator
    estimator.fit(x_train_train, y_train_train, x_train_valid, y_train_valid)
    
    score = scoring(y_train_valid, estimator.predict(x_train_valid))
    print('Score:', score)
    
    return estimator.get_estimator(), score


def fit(estimator, scoring, score_threshold, x_train, y_train, cv):
    
    print('Fit')
    
    trained_estimators = []
    a = 0
    b = 0
    
    for train_idx, valid_idx in cv.split(x_train, y_train):
        
        a += 1
        print('Fit step:', a)
        
        e, score = fit_step(estimator, scoring, x_train, y_train, train_idx, valid_idx)
        
        if score > score_threshold:
            print('OK')
            b += 1
            trained_estimators.append(deepcopy(e))
        else:
            print('Score <= threshold. Skip')
    
    print('Estimators selected:', b, 'from', a)
    
    oof = get_oof(trained_estimators, x_train, y_train)
    print('Final score:', scoring(y_train, oof))

    return oof, trained_estimators


def get_oof(trained_estimators, x_train, y_train):
    
    print('OOF')
    
    oof = np.zeros(x_train.shape[0])
    
    for train_idx, valid_idx in cv.split(x_train, y_train):
        
        # prepare train and validation data
        x_train_train = x_train[train_idx]
        x_train_valid = x_train[valid_idx]

        # collect OOF
        oof_part = predict(trained_estimators, x_train_valid)
        
        oof[valid_idx] = oof_part
    
    return oof


def predict(trained_estimators, x):
    
    print('Predict')
    
    y = np.zeros(x.shape[0])
    
    for estimator in trained_estimators:
        
        y_part = estimator.predict(x)
        
        # average predictions for test data
        y += y_part / len(trained_estimators)
    
    return y


# roc auc metric robust to one class in y_pred

def robust_roc_auc_score(y, y_pred):
    try:
        return roc_auc_score(y, y_pred)
    except:
        return 0.5

robust_roc_auc = make_scorer(robust_roc_auc_score)




#
# Solution
#

random_state=213
np.random.seed(random_state)

# load data
df_train = pd.read_csv('../input/train.csv', index_col='id')
df_test = pd.read_csv('../input/test.csv', index_col='id')

# prepare train data
train_columns = list(df_train.columns)
train_columns.remove('target')
x_train = df_train[train_columns]
y_train = df_train['target']

# scale
temp = RobustScaler().fit_transform(np.concatenate((x_train, df_test), axis=0))
scaled_x_train = temp[:x_train.shape[0]]
scaled_x_test  = temp[x_train.shape[0]:]

# add noise
scaled_x_train += np.random.normal(0, 0.01, scaled_x_train.shape)

# fit

class CustomEstimator(Estimator):
    
    def __init__(self, estimator, param_grid):
        self.estimator = estimator
        self.param_grid = param_grid
    
    def get_estimator(self):
        return self
    
    def fit(self, x_train, y_train, x_valid, y_valid):
        
        # select features
        self.feature_selector = RFECV(self.estimator, min_features_to_select=12, step=15, scoring=robust_roc_auc, cv=20,
                                      verbose=0, n_jobs=-1)
        
        self.feature_selector.fit(x_train, y_train)
        
        print('Features selected:', self.feature_selector.n_features_)
        
        # search for parameters
        self.grid = GridSearchCV(estimator=self.feature_selector.estimator_, param_grid=self.param_grid,
                                 scoring=robust_roc_auc, cv=20, verbose=0, n_jobs=-1)
        
        self.grid.fit(self.feature_selector.transform(x_train), y_train)
        
        print('Best score:', self.grid.best_score_)
        print('Best parameters', self.grid.best_params_)
    
    def predict(self, x):
        return self.grid.best_estimator_.predict(self.feature_selector.transform(x))


cv = StratifiedShuffleSplit(n_splits=20, test_size=0.35, random_state=random_state)

# fit Lasso to use in custom estimator

lasso = Lasso(alpha=0.031, tol=0.01, random_state=random_state, selection='random')

lasso_param_grid = {
    'alpha': [0.019, 0.02, 0.021, 0.022, 0.023, 0.024],
    'tol': [0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016]
}

# lasso_grid = GridSearchCV(estimator=lasso, cv=20, param_grid=lasso_param_grid,
#                           scoring=robust_roc_auc, verbose=0, n_jobs=-1)
#
# lasso_grid.fit(scaled_x_train, y_train)
#
# print('Initial Best Score:', lasso_grid.best_score_)
# print('Initial Best Parameters', lasso_grid.best_params_)

# fit: lasso or lasso_grid.best_estimator_
oof, trained_estimators = fit(CustomEstimator(lasso, lasso_param_grid),
                              r2_score, 0.185, scaled_x_train, y_train.values, cv)

# predict
y = predict(trained_estimators, scaled_x_test)

# save predistions
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = y
submission.to_csv('lasso-submission.csv', index=False)

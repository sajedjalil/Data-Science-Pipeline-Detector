import numpy as np
import pandas as pd

from copy import deepcopy

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score, make_scorer, r2_score

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
        
        print('Best Score:', self.grid.best_score_)
        print('Best Parameters', self.grid.best_params_)
    
    def predict(self, x):
        return self.grid.best_estimator_.predict(x)


def fit_step(estimator, scoring, x_train, y_train, train_idx, valid_idx, oof):
    
    # prepare train and validation data
    x_train_train = x_train[train_idx]
    y_train_train = y_train[train_idx]
    x_train_valid = x_train[valid_idx]
    y_train_valid = y_train[valid_idx]
    
    # fit estimator
    estimator.fit(x_train_train, y_train_train, x_train_valid, y_train_valid)
    
    # collect OOF
    oof_part = estimator.predict(x_train_valid)
    
    score = scoring(y_train_valid, oof_part)
    print('Score:', score)
    
    oof[valid_idx] = oof_part
    
    return estimator.get_estimator(), score


def fit(estimator, scoring, score_threshold, x_train, y_train, cv):
    
    oof = np.zeros(x_train.shape[0])
    
    trained_estimators = []
    
    a = 0
    b = 0
    
    for train_idx, valid_idx in cv.split(x_train, y_train):
        
        a += 1
        
        e, score = fit_step(estimator, scoring, x_train, y_train, train_idx, valid_idx, oof)
        
        if score > score_threshold:
            print('OK')
            b += 1
            trained_estimators.append(deepcopy(e))
        else:
            print('Score <= threshold. Skip')
    
    print(b, 'estimators selected from', a)
    print('Final Score:', scoring(y_train, oof))
    
    return oof, trained_estimators


def predict(trained_estimators, x_test):
    
    y = np.zeros(x_test.shape[0])
    
    for estimator in trained_estimators:
        
        y_part = estimator.predict(x_test)
        
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

df_train = pd.read_csv('../input/train.csv', index_col='id')

df_test = pd.read_csv('../input/test.csv', index_col='id')

train_columns = list(df_train.columns)

train_columns.remove('target')

x_train = df_train[train_columns]
y_train = df_train['target']

# scale

temp = RobustScaler().fit_transform(np.concatenate((x_train, df_test), axis=0))

scaled_x_train = temp[:x_train.shape[0]]

scaled_x_test = temp[x_train.shape[0]:]

# fit

cv = StratifiedShuffleSplit(n_splits=20, test_size=0.33, random_state=42)

lasso = Lasso(random_state=42, selection='random')

lasso_param_grid = {
    'alpha': [0.02, 0.0225, 0.025, 0.0275, 0.03, 0.0325, 0.035, 0.0375, 0.04],
    'tol': [0.00025, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002]
}

oof, trained_estimators = fit(GridSearchCvEstimator(lasso, lasso_param_grid, robust_roc_auc, 20),
                              robust_roc_auc_score, 0.77, scaled_x_train, y_train.values, cv)

# predict

y = predict(trained_estimators, scaled_x_test)

# save predistions

submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = y

submission.to_csv('lasso-submission.csv', index=False)

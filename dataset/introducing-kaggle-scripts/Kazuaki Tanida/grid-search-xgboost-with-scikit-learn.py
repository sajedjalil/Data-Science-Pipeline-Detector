#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Check this gist for xgboost wrapper: https://gist.github.com/slaypni/b95cb69fd1c82ca4c2ff
 
import sys
import math
 
import numpy as np
from sklearn.grid_search import GridSearchCV
 
sys.path.append('xgboost/wrapper/')
import xgboost as xgb
 
 
class XGBoostClassifier():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})
 
    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)
 
    def predict(self, X):
        num2label = {i: label for label, i in self.label2num.items()}
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])
 
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)
 
    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)
 
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self
    
    
def logloss(y_true, Y_pred):
    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in zip(Y_pred, y_true)) / len(Y_pred)


def main():
    clf = XGBoostClassifier(
        eval_metric = 'auc',
        num_class = 2,
        nthread = 4,
        silent = 1,
        )
    parameters = {
        'num_boost_round': [100, 250, 500],
        'eta': [0.05, 0.1, 0.3],
        'max_depth': [6, 9, 12],
        'subsample': [0.9, 1.0],
        'colsample_bytree': [0.9, 1.0],
    }
    clf = GridSearchCV(clf, parameters, n_jobs=1, cv=2)
    
    clf.fit([[1,2], [3,4], [2,1], [4,3], [1,0], [4,5]], ['a', 'b', 'a', 'b', 'a', 'b'])
    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print('score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    print('predicted:', clf.predict([[1,1]]))


if __name__ == '__main__':
    main()
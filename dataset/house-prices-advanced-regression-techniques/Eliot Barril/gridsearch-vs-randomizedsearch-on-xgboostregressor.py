# Inspired from https://www.kaggle.com/tanitter/introducing-kaggle-scripts/grid-search-xgboost-with-scikit-learn/run/23363/code
import sys
import math
 
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time
from scipy.stats import skew
import pandas as pd


TARGET = 'SalePrice'
NROWS = None
SEED = 0


## Load the data ##
train = pd.read_csv("../input/train.csv")

ntrain = train.shape[0]

## Preprocessing ##

y_train = np.log(train[TARGET]+1)


train.drop([TARGET], axis=1, inplace=True)


all_data =train.loc[:,'MSSubClass':'SaleCondition']

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:

x_train = np.array(all_data[:train.shape[0]])

x_fit,x_test,y_fit,y_test = train_test_split(x_train, y_train, train_size =0.33, 
                            random_state=SEED)
 
sys.path.append('xgboost/wrapper/')
import xgboost as xgb
 
 
class XGBoostRegressor():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'reg:linear'})
 
    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        dtrain = xgb.DMatrix(X, label=y)
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)
 
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)
 
    def get_params(self, deep=True):
        return self.params
        
    def score(self, X, y):
        Y = self.predict(X)
        return np.sqrt(mean_squared_error(y, Y))
 
    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self
    

def main():
    clf = XGBoostRegressor(
        eval_metric = 'rmse',
        nthread = 4,
        eta = 0.1,
        num_boost_round = 80,
        max_depth = 5,
        subsample = 0.5,
        colsample_bytree = 1.0,
        silent = 1,
        )
    parameters = {
        'num_boost_round': [10, 25, 50],
        'eta': [0.05, 0.1, 0.3],
        'max_depth': [3, 4, 5],
        'subsample': [0.9, 1.0],
        'colsample_bytree': [0.9, 1.0],
    }
    
    clf1 = GridSearchCV(clf, parameters, n_jobs=1, cv=2)
    clf2 = RandomizedSearchCV(clf, parameters, n_jobs=1, cv=2)
    
    start1 = time.time()
    clf1.fit(x_fit, y_fit)
    best_parameters, score, _ = max(clf1.grid_scores_, key=lambda x: x[1])
    print('GridSearchCV Results: ')
    print(score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    end1 = time.time()
    print('time elapsed: ' + str(end1-start1))
    y_predict1 = clf1.predict(x_test)
    print('')
    
    start2 = time.time()
    clf2.fit(x_fit, y_fit)
    best_parameters, score, _ = max(clf2.grid_scores_, key=lambda x: x[1])
    print('RandomizedSearchCV Results: ')
    print(score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    end2 = time.time()
    print('time elapsed: ' + str(end2-start2))            
    y_predict2 = clf2.predict(x_test)


if __name__ == '__main__':
    main()


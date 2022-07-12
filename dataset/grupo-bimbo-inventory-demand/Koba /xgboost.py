# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
import xgboost as xgb

# custom scroring function
def rmsle_func(truths, preds):
    truths = np.asarray(truths)
    preds = np.asarray(preds)
    
    n = len(truths)
    diff = (np.log(preds+1) - np.log(truths+1))**2
    print(diff, n, np.sum(diff))
    return np.sqrt(np.sum(diff)/n)

# read data in
train = pd.read_csv('../input/train.csv', nrows=7000000)
test = pd.read_csv('../input/test.csv')

print('Train subset shape: ', train.shape)
print('Train head\n',train.iloc[1:6,:])
print('\nTest head\n',test.iloc[1:6,:])

ids = test['id']
test = test.drop(['id'],axis = 1)

train = train.loc[train['Demanda_uni_equil'] < 51,:]


y = train['Demanda_uni_equil']
X = train[test.columns.values]

print(X.shape, y.shape)

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1729)
print(X_train.shape, X_test.shape)

# logistic classifier from xgboost
rmsle  = make_scorer(rmsle_func, greater_is_better=False)

xlf = xgb.XGBRegressor(objective="reg:linear", seed=1729)
xlf.fit(X_train, y_train, eval_metric='rmse', verbose = True, eval_set = [(X_test, y_test)])

# calculate the auc score
preds = xlf.predict(X_test)

print('\nMean Square error" ', mean_squared_error(y_test,preds))

# submission
test_preds = np.around(xlf.predict(test), decimals=1)

submission = pd.DataFrame({"id":ids, "Demanda_uni_equil": test_preds})
submission.to_csv("submission.csv", index=False)
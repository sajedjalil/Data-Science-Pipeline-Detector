# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random

import xgboost as xgb # XGBoost implementation



from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

random.seed(1)

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

features = [x for x in train.columns if x not in ['id','loss']]
#print(features)

cat_features = [x for x in train.select_dtypes(include=['object']).columns if x not in ['id','loss']]
num_features = [x for x in train.select_dtypes(exclude=['object']).columns if x not in ['id','loss']]

from scipy.stats import norm, lognorm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

train['log_loss'] = np.log(train['loss'])

ntrain = train.shape[0]
ntest = test.shape[0]
train_test = pd.concat((train[features], test[features])).reset_index(drop=True)
for c in range(len(cat_features)):
    train_test[cat_features[c]] = train_test[cat_features[c]].astype('category').cat.codes

train_x = train_test.iloc[:ntrain,:]
test_x = train_test.iloc[ntrain:,:]


xgdmat = xgb.DMatrix(train_x, train['log_loss']) # Create our DMatrix to make XGBoost more efficient

    
    
params = {'eta': 0.1, 'seed':0, 'subsample': 0.7, 'colsample_bytree': 0.3, 
             'objective': 'reg:linear', 'max_depth':7, 'min_child_weight':3,'alpha':0.01,'gamma':0.5}
num_rounds = 500
bst = xgb.train(params, xgdmat, num_boost_round = num_rounds)
'''
res=xgb.cv(params,xgdmat,num_boost_round = num_rounds,nfold=3,seed=0,stratified=True,early_stopping_rounds=10)
print(res)

cv_mean=res.iloc[-1,0]
cv_std=res.iloc[-1,1]

print(cv_mean)
print(cv_std)
'''
test_xgb = xgb.DMatrix(test_x)
submission = pd.read_csv("../input/sample_submission.csv")
submission.iloc[:, 1] = np.exp(bst.predict(test_xgb))
submission.to_csv('xgb_starter.sub.csv', index=None)
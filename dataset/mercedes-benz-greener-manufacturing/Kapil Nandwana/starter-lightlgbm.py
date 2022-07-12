# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import lightgbm as lgb
import gc

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        
for c in test.columns:
    if test[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(test[c].values)) 
        test[c] = lbl.transform(list(test[c].values))
        
y_train = train["y"]
train.drop('y', axis=1, inplace=True)

import xgboost as xgb

"""xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}"""

dtrain = lgb.Dataset(train, label=y_train)
dtest = lgb.Dataset(test)

#dtrain = xgb.DMatrix(train, y_train)
#dtest = xgb.DMatrix(test)

params = {}
params['max_bin'] = 9
params['learning_rate'] = 0.1 # shrinkage_rate
params['boosting_type'] = 'dart'
params['objective'] = 'regression'
params['metric'] = 'l2'          # or 'mae'
params['sub_feature'] = 0.5      # feature_fraction 
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 80
params['num_leaves'] = 500       # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf

watchlist = [dtest]
model = lgb.train(params, dtrain, 490, watchlist)


# Uncomment to tune XGB `num_boost_rounds`

#cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
#   verbose_eval=True, show_stdv=False)

#num_boost_rounds = len(cv_result)
#print(num_boost_rounds)
# num_boost_round = 489

#model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_pred = model.predict(test)
output = pd.DataFrame({'id': test['ID'], 'y': y_pred})
output.to_csv('submit1.csv', index=False)

























# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.metrics import r2_score, explained_variance_score, make_scorer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train2=train
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

y_train = train["y"]

#y_train=np.log((train["y"]))
y_mean = np.mean(y_train)

train.drop('y', axis=1, inplace=True)

import xgboost as xgb
xgb_params = {
    'eta': 0.01,
    'max_depth': 8,
    'subsample': 0.85,
    'colsample_bytree': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(train, y_train)
dtest = xgb.DMatrix(test)


# Uncomment to tune XGB `num_boost_rounds`

#cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=2000, early_stopping_rounds=50,
#   verbose_eval=True, show_stdv=False)

#num_boost_rounds = len(cv_result)
#print(num_boost_rounds)
# num_boost_round = 489
#num_boost_round = 243

#model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)


model2=GradientBoostingRegressor(loss='ls', learning_rate=0.04, n_estimators=500, 
        max_depth=6, verbose=1)

#model2.fit(train, y_train)
#y_pred=(model2.predict(test))
r2 = make_scorer(explained_variance_score)


scores=cross_val_score(model2, train, y_train, cv=5, n_jobs=-1, scoring=r2)
print(scores) 

#y_pred = np.exp((model.predict(dtest)))
#output = pd.DataFrame({'id': test['ID'], 'y': y_pred})
#output.to_csv('submit1.csv', index=False)
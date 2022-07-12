# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from sklearn.metrics import r2_score
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#Lol... I cant download the CSV... lets run it again
def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

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
xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(train, y_train)
dtest = xgb.DMatrix(test)


# Uncomment to tune XGB `num_boost_rounds`

cv_result = xgb.cv(xgb_params, dtrain, nfold=10,num_boost_round=1000, early_stopping_rounds=20,
   verbose_eval=True, show_stdv=False, feval=xgb_r2_score)

#num_boost_rounds = len(cv_result)
num_boost_rounds = int((cv_result.shape[0] - 20) / (1 - 1 / 10))

print(num_boost_rounds)
# num_boost_round = 489

model1 = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
model2 = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
model3 = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
model4 = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
model5 = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds+10)


y_pred1 = model1.predict(dtest)
y_pred2 = model2.predict(dtest)
y_pred3 = model3.predict(dtest)
y_pred4 = model4.predict(dtest)
y_pred5 = model5.predict(dtest)

y_pred = (y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5)/5
output = pd.DataFrame({'id': test['ID'], 'y': y_pred})
output.to_csv('submit1.csv', index=False)
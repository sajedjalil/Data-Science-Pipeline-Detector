import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import xgboost as xgb

import datetime



v_train = pd.read_csv('../input/train.csv')
v_test = pd.read_csv('../input/test.csv')
v_macro = pd.read_csv('../input/macro.csv')
id_test = v_test.id


train=pd.concat([v_train,v_macro],axis=1)        
test=pd.concat([v_test,v_macro],axis=1)           



y_train = train["price_doc"]
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)



for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))
        #x_train.drop(c,axis=1,inplace=True)
        
for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))
        #x_test.drop(c,axis=1,inplace=True)
        

x_train.fillna(x_train.median(),inplace=True)
x_test.fillna(x_test.median(),inplace=True)

b=SelectKBest(f_regression, k=250) #k is number of features.
b.fit(x_train, y_train)
    #print b.get_params

idxs_selected = b.get_support(indices=True)
X_train=x_train[idxs_selected]

X_test=x_test[idxs_selected]
xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
   'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}



dtrain = xgb.DMatrix(X_train,y_train)
dtest = xgb.DMatrix(X_test)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})


output.to_csv('xgbParamTunSub24052017.csv', index=False)

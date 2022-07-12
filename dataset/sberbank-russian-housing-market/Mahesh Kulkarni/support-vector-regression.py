import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
from sklearn.svm import SVR

import datetime



train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
macro = pd.read_csv('../input/macro.csv')
id_test = test.id

#clean data        
        



y_train = train["price_doc"]
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)



x_train.fillna(x_train.mean(),inplace=True)
x_test.fillna(x_test.mean(),inplace=True)




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
        



#xgb_params = {
#    'eta': 0.05,
#    'max_depth': 5,
#    'subsample': 0.7,
#    'colsample_bytree': 0.7,
#    'objective': 'reg:linear',
#    'eval_metric': 'rmse',
#    'silent': 1
#}



#dtrain = xgb.DMatrix(x_train, y_train)
#dtest = xgb.DMatrix(x_test)

#cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
#    verbose_eval=50, show_stdv=False)

#num_boost_rounds = len(cv_output)
#model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)
model = SVR(kernel='rbf', C=1000, gamma=100)
model.fit(x_train,y_train)
y_predict = model.predict(x_test)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})


output.to_csv('SVM22052017.csv', index=False)



#can't merge train with test because the kernel run f
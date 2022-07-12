import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
#now = datetime.datetime.now()
#Special thanks to Reynaldo

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
macro = pd.read_csv('../input/macro.csv')
id_test = test.id
train.sample(3)
# Any results you write to the current directory are saved as output.
'''
train["timestamp"] = pd.to_datetime(train["timestamp"])
train["year"], train["month"], train["day"] = train["timestamp"].dt.year,train["timestamp"].dt.month,train["timestamp"].dt.day

test["timestamp"] = pd.to_datetime(test["timestamp"])
test["year"], test["month"], test["day"] = test["timestamp"].dt.year,test["timestamp"].dt.month,test["timestamp"].dt.day
'''

#feauture engineering
train["year_old"] = 2020-train["build_year"]
test["year_old"] = 2020-test["build_year"]

train["floor_inverse"] = train["max_floor"]-train["floor"]
test["floor_inverse"] = test["max_floor"]-test["floor"]

train["area/room"] = train["full_sq"]/train["num_room"]
test["area/room"] = test["full_sq"]/test["num_room"]

train["non_res/res"] = train["full_sq"]-train["life_sq"]/train["life_sq"]
test["non_res/res"] = test["full_sq"]-test["life_sq"]/test["life_sq"]


y_train = train["price_doc"]
x_train = train.drop(["id","timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

#can't merge train with test because the kernel run for very long time
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
        
xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)


cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
print(len(cv_output))
num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.head()

output.to_csv('xgbSub.csv', index=False)
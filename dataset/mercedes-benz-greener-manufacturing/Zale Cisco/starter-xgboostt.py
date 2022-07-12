import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))


y_train = train["y"]
y_mean = np.mean(y_train)
train.drop('y', axis=1, inplace=True)


import xgboost as xgb
xgb_params = {
    'eta': 0.02, 
    'max_depth': 4,
    'subsample': 0.95 ,
    # 'colsample_bytree': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(train, y_train)
dtest = xgb.DMatrix(test)


# Uncomment to tune XGB `num_boost_rounds`

cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
   verbose_eval=True, show_stdv=False)

num_boost_rounds = len(cv_result)
print(num_boost_rounds)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_pred = model.predict(dtest)
output = pd.DataFrame({'id': test['ID'], 'y': y_pred})
output.to_csv('submit1.csv', index=False)
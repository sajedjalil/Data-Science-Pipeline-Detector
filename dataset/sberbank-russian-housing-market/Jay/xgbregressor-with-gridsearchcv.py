import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import datetime
from sklearn.model_selection import GridSearchCV

now = datetime.datetime.now()

# Load the data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
macro = pd.read_csv('../input/macro.csv')
id_test = test.id
train.sample(3)

y_train_full = train['price_doc']
x_train_full = train.drop(["id", "timestamp", "price_doc"], axis=1)

x_test = test.drop(["id", "timestamp"], axis=1)

# Convert columns that are not numeric to a numeric value
for c in x_train_full.columns:
    if x_train_full[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train_full[c].values))
        x_train_full[c] = lbl.transform(list(x_train_full[c].values))
        # x_train_full.drop(c,axis=1,inplace=True)

for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values))
        x_test[c] = lbl.transform(list(x_test[c].values))
        # x_test.drop(c,axis=1,inplace=True)

# Various hyper-parameters to tune
xgb1 = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(x_train_full,
         y_train_full)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

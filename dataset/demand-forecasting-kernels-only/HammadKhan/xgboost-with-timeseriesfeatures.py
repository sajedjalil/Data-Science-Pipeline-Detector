import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import lightgbm as lgb
from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import KFold,train_test_split,TimeSeriesSplit
import catboost as cgb
import xgboost as xgb
import warnings


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample = pd.read_csv('../input/sample_submission.csv')
print("Train rows and columns : ", train.shape)
print("Test rows and columns : ", test.shape)


df = pd.concat([train,test])
print(df.shape)
df.head()

df['date'] = pd.to_datetime(df['date'],infer_datetime_format=True)
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['week_of_year']  = df.date.dt.weekofyear


df["median-store_item-month"] = df.groupby(['month',"item","store"])["sales"].transform("median")
df["mean-store_item-week"] = df.groupby(['week_of_year',"item","store"])["sales"].transform("mean")
df["item-month-sum"] = df.groupby(['month',"item"])["sales"].transform("sum") # total sales of that item  for all stores

df["store-month-sum"] = df.groupby(['month',"store"])["sales"].transform("sum") # total sales of that store  for all items

df['store_item_shifted-365'] = df.groupby(["item","store"])['sales'].transform(lambda x:x.shift(365)) # sales for that 1 year  ago
df["item-week_shifted-90"] = df.groupby(['week_of_year',"item"])["sales"].transform(lambda x:x.shift(12).sum()) # shifted total sales for that item 12 weeks (3 months) ago

df['store_item_shifted-365'].fillna(df['store_item_shifted-365'].mode()[0], inplace=True)

del df['month']
del df['week_of_year']
df.isnull().sum()

col = [i for i in df.columns if i not in ['date','id']]
y = 'sales'

train = df.loc[~df.sales.isna()]
print("new train",train.shape)
test = df.loc[df.sales.isna()]
print("new test",test.shape)

X_train = train.drop(['date','sales','id'], axis=1)
y_train = train['sales'].values
X_test = test.drop(['id','date','sales'], axis=1)


train_x, train_cv, y, y_cv = train_test_split(train[col],train[y], test_size=0.2, random_state=2018)

def run_xgb(train_X, train_y, val_X, val_y):
    params = {
             'colsample_bytree': 0.67,
             'gamma': 0.19,
             'learning_rate': 0.1,
             'max_depth': 6,
             'eval_metric' : 'mae',
             'min_child_weight': 3,
             'nthread': 5,
             'objective': 'reg:linear',
             'scale_pos_weight': 1,
             'subsample': 0.60,
             'random_state': 2018,
             'n_estimators': 400,
             'reg_alpha': 1.92,
             'reg_lambda': 6.28
             }
    #plst = list(param.items())    
    tr_data = xgb.DMatrix(train_X, train_y)
    va_data = xgb.DMatrix(val_X, val_y)
    watchlist = [(tr_data, 'train'), (va_data, 'valid')]
    model_xgb = xgb.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds = 50, verbose_eval=10)
    return model_xgb
    
model =  run_xgb(train_X = train_x, train_y = y, val_X = train_cv, val_y = y_cv)
y_test = model.predict(xgb.DMatrix(test[col]), num_iteration=model.best_iteration)


sample['sales'] = y_test
sample.to_csv('Submission_LGBM.csv', index=False)





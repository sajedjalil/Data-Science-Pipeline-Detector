import pandas as pd
from pandas import Series,DataFrame

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import operator

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
import xgboost as xgb

types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(float),
         'PromoInterval': np.dtype(str)}

print("Reading in files")
train_df = pd.read_csv("../input/train.csv", parse_dates=[2], dtype=types, low_memory=False)
test_df = pd.read_csv("../input/test.csv", parse_dates=[3], dtype=types, low_memory=False)
store_df = pd.read_csv("../input/store.csv", low_memory=False)

# examine the first few records
# train_df.head()
# store_df.head()

# what types do we have
# train_df.info()

# check correlations
# train_df.corr()

# As expected Sales are positively correlated with stored being open and the number of Customers.  
# Sales are negatively correlated with the Day of the Week.
# The Day of the Week is negatively correlated with the store being open and slightly with a Promo occurring. 

# summary stats
# train_df.describe().transpose()

# test_df.head()

# merge the store information with the training data
print("Merging training and store data")
train_df = pd.merge(train_df, store_df, on='Store', how='left')

# summary stats
print("Summary Stats - Training Data")
train_df.describe()
# Looks like we have some stores with no Sales, some stores closed

# merge the store information with the test data
print("Merging test and store data")
test_df = pd.merge(test_df, store_df, on='Store', how='left')

# summary stats
print("Summary Stats - Test Data")
test_df.describe()

# preprocessing
# if no info provided, assume the store is open
train_df.fillna(1, inplace=True)
test_df.fillna(1, inplace=True)

# use only open stores
train_df = train_df[train_df["Open"] != 0]

# use only stores with positive sales
train_df = train_df[train_df["Sales"] > 0]

# more preprocessing
mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
train_df.StoreType.replace(mappings, inplace=True)
train_df.Assortment.replace(mappings, inplace=True)
train_df.StateHoliday.replace(mappings, inplace=True)

train_df['Year'] = train_df.Date.dt.year
train_df['Month'] = train_df.Date.dt.month
train_df['Day'] = train_df.Date.dt.day
train_df['DayOfWeek'] = train_df.Date.dt.dayofweek
train_df['WeekOfYear'] = train_df.Date.dt.weekofyear

test_df.StoreType.replace(mappings, inplace=True)
test_df.Assortment.replace(mappings, inplace=True)
test_df.StateHoliday.replace(mappings, inplace=True)

test_df['Year'] = test_df.Date.dt.year
test_df['Month'] = test_df.Date.dt.month
test_df['Day'] = test_df.Date.dt.day
test_df['DayOfWeek'] = test_df.Date.dt.dayofweek
test_df['WeekOfYear'] = test_df.Date.dt.weekofyear

# checking for NaN values
test_df.isnull().sum().sum()
test_df.isnull().any().any()

# Source: https://www.kaggle.com/cast42/rossmann-store-sales/xgboost-in-python-with-rmspe-v2
def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

# Source: https://www.kaggle.com/cast42/rossmann-store-sales/xgboost-in-python-with-rmspe-v2
def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

features = ['Store','StoreType','Assortment','StateHoliday','DayOfWeek','Month','Day','Year','WeekOfYear','SchoolHoliday','Promo','Promo2','CompetitionDistance']
print("Selected Features: {0}".format(features))

xgb_params = {"objective": "reg:linear",
              "booster": "gbtree",
              "eta": 0.03,     # small shrinkage
              "max_depth": 7,
              "subsample": 0.7,   # to prevent overfitting and add randomness
              "seed": 1, 
              "silent": 1}

num_rounds = 500


# Train XGBoost model
print("Training the model")
X_train, X_valid = train_test_split(train_df, test_size=0.012, random_state=10)
y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
                                 
gbm = xgb.train(xgb_params, dtrain, num_rounds, evals=watchlist, early_stopping_rounds=10, feval=rmspe_xg, verbose_eval=True)       

# Validate
print("Validating")
yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
error = rmspe(X_valid.Sales.values, np.expm1(yhat))
print('RMSPE: {0}'.format(error))

# Make predictions using the test set
print("Making predictions")
dtest = xgb.DMatrix(test_df[features])
ypred = gbm.predict(dtest)

# create submission file
print("Creating submission file")
result = pd.DataFrame({"Id": test_df["Id"], "Sales": np.expm1(ypred)})
result.to_csv("xgboost_submission.csv", index=False)
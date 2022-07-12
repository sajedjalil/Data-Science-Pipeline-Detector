#!/usr/bin/python

'''
Based on https://www.kaggle.com/justdoit/rossmann-store-sales/xgboost-in-python-with-rmspe/code
Public Score :   0.10566
Private Validation Score : 0.099804
'''

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb

# Gather some features
def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear'])

    # add some more with a bit of preprocessing
    features.append('SchoolHoliday')
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)

    features.append('StateHoliday')
    stateHoliday_mapping = {label:float(idx) for idx,label in enumerate(data['StateHoliday'].unique())}
    data['StateHoliday'] = data['StateHoliday'].map(stateHoliday_mapping)
    features.append('DayOfWeek')
    features.append('month')
    features.append('day')
    features.append('year')
    data['year'] = data.Date.dt.year
    data['month'] = data.Date.dt.month
    data['day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek

    # create for dummy variables using get_dummies, then exclude the first dummy column
    StoreType_dummies = pd.get_dummies(data.StoreType, prefix='StoreType').iloc[:, 1:]
    StoreType_dummies.columns = StoreType_dummies.columns.str.replace('_','') # Workaround for bug in Xgboost
    data = pd.concat([data, StoreType_dummies], axis=1)

    # create for dummy variables using get_dummies, then exclude the first dummy column
    Assortment_dummies = pd.get_dummies(data.Assortment, prefix='Assortment').iloc[:, 1:]
    Assortment_dummies.columns = Assortment_dummies.columns.str.replace('_','') # Workaround for bug in Xgboost
    data = pd.concat([data, Assortment_dummies], axis=1)

    PromoInterval_dummies = pd.get_dummies(data.PromoInterval, prefix='PromoInterval').iloc[:, 1:]
    PromoInterval_dummies.columns = PromoInterval_dummies.columns.str.replace(r'[_,]','') # Workaround for bug in Xgboost
    data = pd.concat([data, PromoInterval_dummies], axis=1)

    newCol = list(StoreType_dummies.columns) + list(Assortment_dummies.columns) + list(PromoInterval_dummies.columns)
    features.extend(newCol)
    return data

## Start of main script

print("Load the training, test and store data using pandas")
train = pd.read_csv("../input/train.csv", parse_dates=[2],dtype={'StateHoliday':str})
test = pd.read_csv("../input/test.csv", parse_dates=[3])
store = pd.read_csv("../input/store.csv")

print("Assume store open, if not provided")
test.fillna(1, inplace=True)

print("Consider only open stores for training. Closed stores wont count into the score.")
train = train[train["Open"] != 0]
print("Use only Sales bigger then zero")
train = train[train["Sales"] > 0]

print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features = []

print("augment features")
train = build_features(features, train)
test = build_features([], test)
print(features)

print('training data processed')

def rmspe(y, yhat):
    return np.sqrt(np.mean(((y - yhat)/y) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

print("Train xgboost model")
X = train[features]
y = np.log1p(train["Sales"])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=10000,random_state=1302)

xgb_model = xgb.XGBRegressor(n_estimators=300,
    max_depth=11,
    learning_rate=0.3,
    subsample=0.8,
    colsample_bytree=0.6,
    ).fit(X_train,y_train,eval_metric=rmspe_xg,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=100)

yhat = xgb_model.predict(X_test)
error = rmspe(np.expm1(y_test), np.expm1(yhat))
print ('RMSPE on test holdout: {:.6f}'.format(error))

print("Make predictions on the test set")
test_probs = xgb_model.predict(test[features])

# Make Submission
result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(test_probs)})
result.to_csv("xgboost_26_submission.csv", index=False)


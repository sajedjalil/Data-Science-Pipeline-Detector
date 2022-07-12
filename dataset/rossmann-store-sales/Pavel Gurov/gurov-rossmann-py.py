import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model.logistic import LogisticRegression
from datetime import datetime
from sklearn import cross_validation

import xgboost as xgbb

def dataPrepare(data):
    data['StateHoliday'] = data['StateHoliday'].map({
        '0': 0,
        'a': 1,
        'b': 2,
        'c': 3,
        0: 0
    }).astype(int)

    data['Open'] = data['Open'].fillna(1) #CHECK
    return data

train_file = '../input/train.csv'
test_file = '../input/test.csv'
output_file = 'predictions.csv'

train = pd.read_csv(train_file, index_col='Date', parse_dates=['Date'])
train['Date'] = train.index
train['Week'] = train.index.week
train['Year'] = train.index.year
train['Month'] = train.index.month
train['Quarter'] = train.index.quarter
train = dataPrepare(train)

test = pd.read_csv(test_file, parse_dates=['Date'])

test = dataPrepare(test)
test['Week'] = test.Date.dt.week
test['Year'] = test.Date.dt.year
test['Month'] = test.Date.dt.month
test['Quarter'] = test.Date.dt.quarter

stories = [1,3,7,8,9]#set(test.Store.values)
stories = set(test.Store.values)
predictors = [
    'DayOfWeek',
    # 'Date',
    'Week',
    'Year',
    'Month',
    'Quarter',
    # 'Sales2',
    # 'Customers',
    'Open',
    'Promo',
    'StateHoliday',
    'SchoolHoliday'
]

params = {"objective": "reg:linear",
    "eta": 0.02,
    "booster": "gbtree",
    "max_depth": 10,
     "min_child_weight": 15,
    "silent": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.7}
num_trees=1000

def saleCode(x, division):
    for i, value in enumerate(division[:-1]):
        if value - 1 < x < division[i + 1] + 1:
            return i
            break


print('Start group')
columns = ['Store', 'DayOfWeek', 'Promo']

medians = train.groupby(columns)['Sales'].median()
medians = medians.reset_index()

test2 = pd.merge(test, medians, on = columns, how = 'left')
assert(len(test2) == len(test))

test2.loc[ test2.Open == 0, 'Sales' ] = 0

test['Sales'] = test2['Sales']
print('Group stories finished')

test.loc[:, 'Sales_rf'] = 0
test.loc[:, 'Sales_xgb'] = 0

for s in stories:

    f1_full = train[train.Store==s]
    f1 = DataFrame(f1_full[f1_full.Open==1])
    count,division = np.histogram(f1.Sales.values, 7)

    storeTrue = {}
    for i, value in enumerate(division[:-1]):
        storeTrue[i] = np.mean([value, division[i+1]])

    f1.loc[:, 'Sales2'] = f1.apply(lambda row: saleCode(row['Sales'], division), axis=1)
    alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=3, min_samples_leaf=3)
    alg.fit(f1[predictors], f1['Sales2'])
    t1_full = test[test.Store==s]
    t1 = t1_full[t1_full.Open==1]
    predictions = alg.predict(t1[predictors])
    test.loc[(test.Store==s) & (test.Open==1), 'Sales_rf'] = list(map(lambda x: storeTrue[x], predictions))
    test.loc[(test.Store==s) & (test.Open==0), 'Sales_rf'] = 0
    print ('Random forest store done: ' + str(s))

    train1 = train[train.Store==s]
    test1 = test[test.Store==s]
    gbm1 = xgbb.train(params, xgbb.DMatrix(train1[predictors], train1["Sales"]), num_trees)
    test_probs = gbm1.predict(xgbb.DMatrix(test.loc[test.Store==s, predictors]))
    test.loc[test.Store==s, 'Sales_xgb'] = test_probs
    print ('XGBoost store done: ' + str(s))


test.loc[test.Open==0, 'Sales_xgb'] = 0

test.loc[:, 'Sales'] = test['Sales']/3.0 + test['Sales_rf']/3.0 + test['Sales_xgb']/3.0

test.loc[test.Open==0, 'Sales'] = 0

test.head(12)
test.describe()

test[[ 'Id', 'Sales' ]].to_csv(output_file, index = False)

print("Up the leaderboard!")

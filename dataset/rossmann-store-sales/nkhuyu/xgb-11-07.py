#!/usr/bin/python

'''
Based on https://www.kaggle.com/justdoit/rossmann-store-sales/xgboost-in-python-with-rmspe/code
Public Score :  not submitted
Private Validation Score :  0.114779
'''

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import operator
import matplotlib
matplotlib.use("Agg") #Needed to save figures
import matplotlib.pyplot as plt

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

def toBinary(featureCol, df):
    values = set(df[featureCol].unique())
    newCol = [featureCol + val for val in values]
    for val in values:
        df[featureCol + val] = df[featureCol].map(lambda x: 1 if x == val else 0)
    return newCol

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
    data.loc[data['StateHoliday'] == 'a', 'StateHoliday'] = '1'
    data.loc[data['StateHoliday'] == 'b', 'StateHoliday'] = '2'
    data.loc[data['StateHoliday'] == 'c', 'StateHoliday'] = '3'
    data['StateHoliday'] = data['StateHoliday'].astype(float)

    features.append('DayOfWeek')
    features.append('month')
    features.append('day')
    features.append('year')
    data['year'] = data.Date.dt.year
    data['month'] = data.Date.dt.month
    data['day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek

    for x in ['a', 'b', 'c', 'd']:
        features.append('StoreType' + x)
        data['StoreType' + x] = data['StoreType'].map(lambda y: 1 if y == x else 0)

    newCol = toBinary('Assortment', data)
    features += newCol

## Start of main script

print("Load the training, test and store data using pandas")
train = pd.read_csv("../input/train.csv", parse_dates=[2])
test = pd.read_csv("../input/test.csv", parse_dates=[3])
store = pd.read_csv("../input/store.csv")

print("Assume store open, if not provided")
train.fillna(1, inplace=True)
test.fillna(1, inplace=True)

print("Consider only open stores for training. Closed stores wont count into the score.")
train = train[train["Open"] != 0]
print("Use only Sales bigger then zero. Simplifies calculation of rmspe")
train = train[train["Sales"] > 200]

print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features = []

print("augment features")
build_features(features, train)
build_features([], test)
print(features)

print('training data processed')

params = {"objective": "reg:linear",
          "eta": 0.02,
          "max_depth": 9,
          "subsample": 0.8,
          "colsample_bytree": 0.8,
          "silent": 1,
          "thread": 1,
          "seed": 1301
          }
num_boost_round = 1500

print("Train a XGBoost model")
X_train, X_valid = train_test_split(train, test_size=0.012, random_state=10)
y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)

watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=200, \
  feval=rmspe_xg, verbose_eval=True)

print("Validating")
yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
error = rmspe(X_valid.Sales.values, np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))

print("Make predictions on the test set")
dtest = xgb.DMatrix(test[features])
test_probs = gbm.predict(dtest)
# Make Submission
result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(test_probs)})
result.to_csv("xgboost_10_submission.csv", index=False)

# XGB feature importances
# Based on https://www.kaggle.com/mmueller/liberty-mutual-group-property-inspection-prediction/xgb-feature-importance-python/code

# ceate_feature_map(features)
# importance = gbm.get_fscore(fmap='xgb.fmap')
# importance = sorted(importance.items(), key=operator.itemgetter(1))

# df = pd.DataFrame(importance, columns=['feature', 'fscore'])
# df['fscore'] = df['fscore'] / df['fscore'].sum()

# featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
# plt.title('XGBoost Feature Importance')
# plt.xlabel('relative importance')
# fig_featp = featp.get_figure()
# fig_featp.savefig('feature_importance_xgb.png',bbox_inches='tight',pad_inches=1)


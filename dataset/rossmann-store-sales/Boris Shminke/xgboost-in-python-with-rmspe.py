import pandas as pd
import numpy as np
from sklearn import cross_validation
import xgboost as xgb

# Gather some features
def build_features(features, data):
    # remove NaNs
    data.fillna(0)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Open', 'Promo', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear'])
    
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
    data['month'] = data.Date.apply(lambda x: x.split('-')[1])
    data['month'] = data['month'].astype(float)
    data['day'] = data.Date.apply(lambda x: x.split('-')[2])
    data['day'] = data['day'].astype(float)

    features.append('StoreType')
    data.loc[data['StoreType'] == 'a', 'StoreType'] = '1'
    data.loc[data['StoreType'] == 'b', 'StoreType'] = '2'
    data.loc[data['StoreType'] == 'c', 'StoreType'] = '3'
    data.loc[data['StoreType'] == 'd', 'StoreType'] = '4'
    data['StoreType'] = data['StoreType'].astype(float)

    features.append('Assortment')
    data.loc[data['Assortment'] == 'a', 'Assortment'] = '1'
    data.loc[data['Assortment'] == 'b', 'Assortment'] = '2'
    data.loc[data['Assortment'] == 'c', 'Assortment'] = '3'
    data['Assortment'] = data['Assortment'].astype(float)

print("Load the training, test and store data using pandas")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
store = pd.read_csv("../input/store.csv")

print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features = []

print("augment features")
build_features(features, train)
build_features([], test)

params = {"eta": 1000,
          "max_depth": 3,
          "silent": 1,
          "subsample": 0.5,
          "colsample_bytree": 0.5,
          "seed": 1
          }
num_round = 5

def custobj(preds, dtrain):
    labels = dtrain.get_label()
    grad = [(y - yh) / y ** 2 if y != 0 else 0 for (yh, y) in zip(preds, labels)]
    hess = [1. / y ** 2 if y != 0 else 0 for y in labels]
    return grad, hess

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'RMSPE', np.sqrt(np.average([(1 - yh/y) ** 2 for (yh, y) in zip(preds, labels) if y != 0]))

print("Train a XGBoost model")
dtrain = xgb.DMatrix(train[features], train["Sales"])
xgb.cv(params, dtrain, num_round, nfold=3, seed = 0, obj = custobj, feval=evalerror)

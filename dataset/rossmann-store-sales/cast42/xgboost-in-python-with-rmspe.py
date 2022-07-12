import pandas as pd
import numpy as np
from sklearn import cross_validation
import xgboost as xgb


# Thanks to Chenglong Chen for providing this in the forum
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w


def ToZero(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y > 0
    w[ind] = y[ind]
    return w


def rmspe(y, yhat):
    y = y.values
    w = ToWeight(y)
    yhat = ToZero(yhat)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe


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

params = {"objective": "reg:linear",
          "eta": 0.25,
          "max_depth": 8,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1,
          "booster": "gbtree"}
num_trees = 500

X_train, X_test = cross_validation.train_test_split(train)

print("Train a XGBoost model")
dtrain = xgb.DMatrix(X_train[features], np.log(X_train["Sales"] + 1))
dtest = xgb.DMatrix(X_test[features])
gbm = xgb.train(params, dtrain, num_trees)

print("Validating")
train_probs = gbm.predict(dtest)
error = rmspe(X_test['Sales'], np.exp(ToZero(train_probs)) - 1)
print('error', error)

print("Make predictions on the test set")
test_probs = gbm.predict(xgb.DMatrix(test[features]))
submission = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(ToZero(test_probs)) - 1})
submission.to_csv("xgboost_submission.csv", index=False)

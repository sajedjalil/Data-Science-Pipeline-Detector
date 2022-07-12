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


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe


def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe


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
    #
    #features.append('StateHoliday')
    #data.loc[data['StateHoliday'] == 'a', 'StateHoliday'] = '1'
    #data.loc[data['StateHoliday'] == 'b', 'StateHoliday'] = '2'
    #data.loc[data['StateHoliday'] == 'c', 'StateHoliday'] = '3'
    #data['StateHoliday'] = data['StateHoliday'].astype(float)

    features.append('DayOfWeek')
    features.append('month')
    features.append('day')
    features.append('year')
    data['year'] = data.Date.apply(lambda x: x.split('-')[0])
    data['year'] = data['year'].astype(float)
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

print("Assume store open, if not provided")
test.fillna(1, inplace=True)

print("Consider only open stores for training. Closed stores wont count into the score.")
train = train[train["Open"] != 0]

print("Add store mean")
sale_means = train.groupby('Store').mean().Sales
sale_means.name = 'SalesMean'

train = train.join(sale_means, on='Store')
test = test.join(sale_means, on='Store')

print("Add customer mean")
customers_means = train.groupby('Store').mean().Customers
customers_means.name = 'CustomersMean'

train = train.join(customers_means, on='Store')
test = test.join(customers_means, on='Store')

print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features = []

print("augment features")
build_features(features, train)
build_features([], test)
print(features)



print("Train a XGBoost model")
val_size = 15000
train = train.sort(['Date'])
print(train.tail(1)['Date'])

params = {"objective": "reg:linear",
          "eta": 0.2,
          "max_depth": 12,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "silent": 1
          }
num_trees = 2000
X_train, X_val = cross_validation.train_test_split(train, test_size=val_size,random_state=1023)
dtrain = xgb.DMatrix(X_train[features], np.log(X_train["Sales"] + 1))
dvalid = xgb.DMatrix(X_val[features], np.log(X_val["Sales"] + 1))
dtest = xgb.DMatrix(test[features])
watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
gbm1 = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)


print("Make predictions on the test set")
test_probs1 = gbm1.predict(xgb.DMatrix(test[features]))
indices1 = test_probs1 < 0
test_probs1[indices1] = 0
test_probs1=np.exp(test_probs1) - 1

submission = pd.DataFrame({"Id": test["Id"], "Sales": test_probs1})
submission.to_csv("xgboost_kscript_submission.csv", index=False)

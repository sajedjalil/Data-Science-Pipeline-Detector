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
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek',
                     'Promo2SinceYear', 'DayOfWeek','SchoolHoliday'])

    # add some more with a bit of preprocessing
    features.append('month')
    features.append('year')

    data['year'] = data['Date'].apply(lambda x: int(x.split('-')[0]))
    data['month'] = data['Date'].apply(lambda x: int(x.split('-')[1]))

    for x in ['StoreType', 'Assortment', 'StateHoliday']:
        features.append(x)
        labels = data[x].unique()
        map_labels = dict(zip(labels, range(0,len(labels))))
        data[x] = data[x].map(map_labels)

print("Load the training, test and store data using pandas")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
store = pd.read_csv("../input/store.csv")

print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

orig_columns = train.columns

train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

print("Consider only open stores with non zero sales for training. Closed stores wont count into the score.")
train = train[train["Open"] != 0]
train = train[train["Sales"] != 0]

features = []

print("augment features")
build_features(features, train)
build_features([], test)
print(features)
print('Removed: {0}'.format(set(orig_columns)-set(features)))

params = {"objective": "reg:linear",
          "eta": 0.3,
          "max_depth": 10,
          "subsample": 0.9,
          "colsample_bytree": 0.7,
          "silent": 1
          }
num_trees = 300

print("Train a XGBoost model")

X_train, X_test = cross_validation.train_test_split(train, test_size=10000, random_state=42)
dtrain = xgb.DMatrix(X_train[features], np.log(X_train["Sales"] + 1))
dvalid = xgb.DMatrix(X_test[features], np.log(X_test["Sales"] + 1))
dtest = xgb.DMatrix(test[features])
watchlist = [(dvalid, 'eval'), (dtrain, 'train')]

gbm = xgb.train(params, dtrain, num_trees,
                evals=watchlist,
                early_stopping_rounds=100, 
                feval=rmspe_xg, verbose_eval=True)

print("Validating")
train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
indices = train_probs < 0
train_probs[indices] = 0
error = rmspe(np.exp(train_probs) - 1, X_test['Sales'].values)
print('error', error)

print("Make predictions on the test set")
test_probs = gbm.predict(xgb.DMatrix(test[features]))

indices = test_probs < 0
test_probs[indices] = 0
closed_idx = (test['Open']==0).values
test_probs[closed_idx] = 0
submission = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(test_probs) - 1})
submission.to_csv("xgboost_submit.csv", index=False)

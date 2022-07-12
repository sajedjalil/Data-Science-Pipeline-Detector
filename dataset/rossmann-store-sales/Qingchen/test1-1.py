import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

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
def build_features(features, data, sort):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['Open', 'Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear'])

    # add some more with a bit of preprocessing
    features.append('SchoolHoliday')
    data['SchoolHoliday'] = data['SchoolHoliday'].astype(float)

    features.append('month')
    features.append('day')
    features.append('year')
    data['year'] = data.Date.apply(lambda x: x.split('-')[0])
    data['year'] = data['year'].astype(float)
    data['month'] = data.Date.apply(lambda x: x.split('-')[1])
    data['month'] = data['month'].astype(float)
    data['day'] = data.Date.apply(lambda x: x.split('-')[2])
    data['day'] = data['day'].astype(float)
    
    features.append('monday')
    features.append('tuesday')
    features.append('wednesday')
    features.append('thursday')
    features.append('friday')
    features.append('saturday')
    features.append('sunday')
    data['monday'] = data.DayOfWeek.apply(lambda x: x == 1)
    data['monday'] = data['monday'].astype(float)
    data['tuesday'] = data.DayOfWeek.apply(lambda x: x == 2)
    data['tuesday'] = data['tuesday'].astype(float)
    data['wednesday'] = data.DayOfWeek.apply(lambda x: x == 3)
    data['wednesday'] = data['wednesday'].astype(float)
    data['thursday'] = data.DayOfWeek.apply(lambda x: x == 4)
    data['thursday'] = data['thursday'].astype(float)
    data['friday'] = data.DayOfWeek.apply(lambda x: x == 5)
    data['friday'] = data['friday'].astype(float)
    data['saturday'] = data.DayOfWeek.apply(lambda x: x == 6)
    data['saturday'] = data['saturday'].astype(float)
    data['sunday'] = data.DayOfWeek.apply(lambda x: x == 7)
    data['sunday'] = data['sunday'].astype(float)

    features.append('days_since')
    data['days_since'] = data.Date.apply(lambda x: (datetime.strptime(x, "%Y-%m-%d") - datetime.strptime("2012-12-31", "%Y-%m-%d")).days)
    data['days_since'] = data['days_since'].astype(float) 

print("Load the training, test and store data using pandas")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train = train[train['Store'] <= 100]
test = test[test['Store'] <= 100]
store = pd.read_csv("../input/store.csv")
store = store[store['Store'] <= 100]
    
print("Assume store open, if not provided")
test.fillna(1, inplace=True)

print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features = []

print("augment features")
build_features(features, train, True)
build_features([], test, True)
print(features)

train_sales = train["Sales"]
train_sales = np.array(train_sales)
test_ids = test["Id"]

train = train[features]
test = test[features]

train = np.array(train)
test = np.array(test) 
          
train = train[:, 1:len(train[0, :])-1]         
test = test[:, 1:len(test[0, :])-1]

# generate interaction terms between day of week and promo
train = np.column_stack([train, np.multiply(train[:, 4], train[:, 12])])
test = np.column_stack([test, np.multiply(test[:, 4], test[:, 12])])
train = np.column_stack([train, np.multiply(train[:, 4], train[:, 13])])
test = np.column_stack([test, np.multiply(test[:, 4], test[:, 13])])
train = np.column_stack([train, np.multiply(train[:, 4], train[:, 14])])
test = np.column_stack([test, np.multiply(test[:, 4], test[:, 14])])
train = np.column_stack([train, np.multiply(train[:, 4], train[:, 15])])
test = np.column_stack([test, np.multiply(test[:, 4], test[:, 15])])
train = np.column_stack([train, np.multiply(train[:, 4], train[:, 16])])
test = np.column_stack([test, np.multiply(test[:, 4], test[:, 16])])
train = np.column_stack([train, np.multiply(train[:, 4], train[:, 17])])
test = np.column_stack([test, np.multiply(test[:, 4], test[:, 17])])
train = np.column_stack([train, np.multiply(train[:, 4], train[:, 18])])
test = np.column_stack([test, np.multiply(test[:, 4], test[:, 18])])

train = np.column_stack([train_sales, train])
train = train[train_sales > 0]

stores = np.unique(test[:, 0])
print(stores)
scores = []
predicted = []
id_list = []
store_output = []
for store in stores:
    s_train = train[:, 1]
    s_test = test[:, 0]
    train_store = train[(s_train == store)]
    test_store = test[s_test == store]
    id_test = test_ids[s_test == store]
    
    params = {"objective": "reg:linear",
              "eta": 0.007,
              "max_depth": 8,
              "min_child_weight": 10,
              "subsample": 0.7,
              "colsample_bytree": 0.6,
              "silent": 1
              }
    num_trees = 1000
    print("Train a XGBoost model")
    X_train = train_store
    dtrain = xgb.DMatrix(X_train[:, 1::], np.log(X_train[:, 0] + 1))
    gbm = xgb.train(params, dtrain, num_trees, feval=rmspe_xg)
     
    #print("Make predictions on the test set")
    test_probs = gbm.predict(xgb.DMatrix(test_store))
    indices = test_probs < 0
    test_probs[indices] = 0
    predicted = predicted + test_probs.tolist()
    id_list = id_list + id_test.tolist()
    store_output = store_output + (np.ones(len(test_probs))*store).tolist()

predicted = np.array(predicted)
id_list = np.array(id_list)
submission = pd.DataFrame({"Id": id_list, "Sales": np.exp(predicted) - 1})  
submission.to_csv("output1.csv", index=False)

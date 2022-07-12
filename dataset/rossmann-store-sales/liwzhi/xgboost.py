import pandas
import numpy
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb


# Thanks to Chenglong Chen for providing this in the forum
def ToWeight(y):
    w = numpy.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = numpy.sqrt(numpy.mean( w * (y - yhat)**2 ))
    return rmspe


def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = numpy.exp(y) - 1
    yhat = numpy.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = numpy.sqrt(numpy.mean(w * (y - yhat)**2))
    return "rmspe", rmspe

TRAIN_DROP_LIST = ['Date', 'Customers']
TEST_DROP_LIST = ['Date', 'Id']
TRAIN_STORE_DROP_LIST = ['Store', 'Sales']
TEST_STORE_DROP_LIST = ['Store']
STORE_DROP_LIST = []
MISSING_VALUE_IMPUTATION = 0
TRAIN_TEST_NEED_ONEHOT_ENCODE = ['DayOfWeek']
STORE_NEED_ONEHOT_ENCODE = ['StoreType', 'Assortment', 'PromoInterval']
TRAIN_TEST_NEED_FACTORIZE = ['StateHoliday']
STORE_NEED_FACTORIZE = ['StoreType', 'Assortment', 'PromoInterval']

# LOAD & PREPROCESS TRAIN SET
train = pandas.read_csv("../input/train.csv", parse_dates=['Date'], low_memory=False)
train['Year'] = [int(x.year) for x in train['Date']]
train['Month'] = [int(x.month) for x in train['Date']]
train['Day'] = [int(x.day) for x in train['Date']]
mean = train.groupby(['Store']).mean()
var = train.groupby(['Store']).std()
train = train.drop(TRAIN_DROP_LIST, axis=1)

test = pandas.read_csv('../input/test.csv', parse_dates=['Date'], low_memory=False)
test['Year'] = [int(x.year) for x in test['Date']]
test['Month'] = [int(x.month) for x in test['Date']]
test['Day'] = [int(x.day) for x in test['Date']]
test_id = test['Id']
test = test.drop(TEST_DROP_LIST, axis=1)

for col in TRAIN_TEST_NEED_FACTORIZE:
    factorized = pandas.factorize(numpy.concatenate([numpy.array(train[col]), numpy.array(test[col])]))[0]
    train[col], test[col] = factorized[: len(train[col])], factorized[len(train[col]):]

train_encoded = []
test_encoded = []
for col in TRAIN_TEST_NEED_ONEHOT_ENCODE:
    # print col
    # print test[col]
    enc = OneHotEncoder(sparse=False)
    train_encoded.append(enc.fit_transform(numpy.array(train[col]).reshape((len(train[col]), 1))))
    train.drop([col], axis=1)
    test_encoded.append(enc.transform(numpy.array(test[col].reshape((len(test[col])), 1))))
    test.drop([col], axis=1)
train_encoded = numpy.concatenate(train_encoded, axis=1)
test_encoded = numpy.concatenate(test_encoded, axis=1)

store = pandas.read_csv("../input/store.csv")
store = store.drop(STORE_DROP_LIST, axis=1)
store['Promo2SinceYear'] = 2016 - store['Promo2SinceYear']
store['CompetitionOpenSinceYear'] = 2016 - store['CompetitionOpenSinceYear']
store['StoreType'] = pandas.factorize(store['StoreType'])[0]
store['Assortment'] = pandas.factorize(store['Assortment'])[0]
# store['Customers'] = mean['Customers']
# store['MSales'] = mean['Sales']
# store['VCustomers'] = var['Customers']
# store['VSales'] = var['Sales']

for col in STORE_NEED_FACTORIZE:
    store[col] = pandas.factorize(store[col])[0]

train_store = pandas.merge(train, store, how='left', on='Store')
train_store['Target'] = train_store['Sales']
train_store = train_store.drop(TRAIN_STORE_DROP_LIST, axis=1)
train_store = train_store.fillna(MISSING_VALUE_IMPUTATION)
test_store = pandas.merge(test, store, how='left', on='Store')
test_store = test_store.drop(TEST_STORE_DROP_LIST, axis=1)
test_store = test_store.fillna(MISSING_VALUE_IMPUTATION)
nonzero_index = numpy.array(test_store['Open']) != 0

train_store_encoded = []
test_store_encoded = []
for col in STORE_NEED_ONEHOT_ENCODE:
    enc = OneHotEncoder(sparse=False)
    if -1 in train_store[col].unique() or -1 in test_store[col].unique():
        train_store[col] += 1
        test_store[col] += 1
        train_store_encoded.append(enc.fit_transform(numpy.array(train_store[col]).reshape((len(train_store[col]), 1))))
        test_store_encoded.append(enc.transform(numpy.array(test_store[col].reshape((len(test_store[col])), 1))))
    else:
        train_store_encoded.append(enc.fit_transform(numpy.array(train_store[col]).reshape((len(train_store[col]), 1))))
        test_store_encoded.append(enc.transform(numpy.array(test_store[col].reshape((len(test_store[col])), 1))))
train_store_encoded = numpy.concatenate(train_store_encoded, axis=1)
train_store_encoded = numpy.concatenate([train_encoded, train_store_encoded], axis=1)
test_store_encoded = numpy.concatenate(test_store_encoded, axis=1)
test_store_encoded = numpy.concatenate([test_encoded, test_store_encoded], axis=1)

train_array = numpy.array(train_store)
test_array = numpy.array(test_store)
train_array = numpy.concatenate([train_store_encoded, train_array], axis=1)
test_array = numpy.concatenate([test_store_encoded, test_array], axis=1)
X = train_array[train_array[:, -1] != 0, :-1]
Y = train_array[train_array[:, -1] != 0, -1]
# print("Assume store open, if not provided")
# test.fillna(1, inumpylace=True)

params = {"objective": "reg:linear",
          'booster':'gbtree',
          "eta": 0.02,
          "max_depth": 10,
          "subsample": 0.9,
          "colsample_bytree": 0.7,
        #  "silent": 1
          }
num_trees = 500

print("Train a XGBoost model")
val_size = 100000
#train = train.sort(['Date'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.01)
#X_train, X_test = train.head(len(train) - val_size), train.tail(val_size)
dtrain = xgb.DMatrix(X_train, numpy.log(y_train + 1))
dvalid = xgb.DMatrix(X_test, numpy.log(y_test + 1))
dtest = xgb.DMatrix(test_array)
watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, feval=rmspe_xg, verbose_eval=True)

print("Validating")
train_probs = gbm.predict(xgb.DMatrix(X_test))
indices = train_probs < 0
train_probs[indices] = 0
error = rmspe(numpy.exp(train_probs) - 1, y_test)
print('error', error)

print("Make predictions on the test set")
test_probs = gbm.predict(xgb.DMatrix(test_array))
indices = test_probs < 0
test_probs[indices] = 0
submission = pandas.DataFrame({"Id": test_id, "Sales": numpy.exp(test_probs) - 0.00001})
submission.to_csv("xgboost_kscript_submission.csv", index=False)

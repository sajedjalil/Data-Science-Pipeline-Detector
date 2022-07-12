import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation
from sklearn.utils import shuffle


def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = np.where(y != 0)[0]
    w[ind] = 1./(y[ind]**2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe


def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.expm1(y)
    yhat = np.expm1(yhat)
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    #mspe = np.mean(w * (y - yhat)**2)
    return "rmspe", rmspe

def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday'])

    # Label encode some features
    features.extend(['StoreType', 'Assortment', 'StateHoliday'])
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)

    features.extend(['DayOfWeek', 'Month', 'Day', 'Year', 'WeekOfYear'])
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.weekofyear

    # CompetionOpen en PromoOpen from https://www.kaggle.com/ananya77041/rossmann-store-sales/randomforestpython/code
    # Calculate time competition open time in months
    features.append('CompetitionOpen')
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
        (data.Month - data.CompetitionOpenSinceMonth)
    # Promo open time in months
    features.append('PromoOpen')
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + \
        (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0

    # Indicate that sales on that day are in promo interval
    features.append('IsPromoMonth')
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    data['monthStr'] = data.Month.map(month2str)
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1

    return data


## Start of main script

print("Load the training, test and store data using pandas")
types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(float),
         'PromoInterval': np.dtype(str)}
train = pd.read_csv("../input/train.csv", parse_dates=[2], dtype=types)
test = pd.read_csv("../input/test.csv", parse_dates=[3], dtype=types)
store = pd.read_csv("../input/store.csv")

print("Assume store open, if not provided")
train.fillna(1, inplace=True)
test.fillna(1, inplace=True)

print("Consider only open stores for training. Closed stores wont count into the score.")
train = train[train["Open"] != 0]
print("Use only Sales bigger then zero. Simplifies calculation of rmspe")
train = train[train["Sales"] > 0]

print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features = []

print("augment features")
build_features(features, train)
build_features([], test)
print(features)

params = {"objective": "reg:linear",
          "eta": 0.1,
          "max_depth": 5,
          "subsample": 0.8,
          "colsample_bytree": 0.7,
          "silent": 1,
          "min_child_weight": 3
          }
num_trees = 50

#trainstores = shuffle(trainstores)
trainstoreData = None
teststoreData = None
totalerror = 0
uniquestores = np.unique(test.Store)
#Remove [:1] for the full run
for i in uniquestores[:1]:
    print('Store:',i)
    individualtrainstores = train[train.Store==i].copy()
    individualteststores = test[test.Store==i].copy()
    individuallasttrainstores = individualtrainstores[(individualtrainstores.Month == 7) &
                                                      (individualtrainstores.Year == 2015)]
    individualtrainstores = individualtrainstores[~((individualtrainstores.Month == 7) &
                                                   (individualtrainstores.Year == 2015))]
    dtrain = xgb.DMatrix(individualtrainstores[features], np.log1p(individualtrainstores["Sales"]))
    dvalid = xgb.DMatrix(individuallasttrainstores[features], np.log1p(individuallasttrainstores["Sales"]))
    watchlist = [(dvalid, 'eval'),(dtrain, 'train')]
    gbm = xgb.train(params, dtrain, num_trees,
                    evals=watchlist,
                    early_stopping_rounds=50,
                    feval=rmspe_xg, verbose_eval=True)
    print("Validating")
    train_probs = gbm.predict(dvalid)
    indices = train_probs < 0
    train_probs[indices] = 0
    error = rmspe(np.expm1(train_probs), individuallasttrainstores['Sales'].values)
    totalerror+=error
    print('error', error)
    individualtrainstoreData = pd.DataFrame({'Predictions': np.expm1(train_probs),
                                             'Sales': individuallasttrainstores["Sales"],
                                             'Store': i })
    if(trainstoreData is None):
        trainstoreData = individualtrainstoreData
    else:
        trainstoreData = pd.concat((trainstoreData,individualtrainstoreData))
    del dtrain
    dtrain = None
    del dvalid
    dvalid = None
    gc.collect()
    dtest = xgb.DMatrix(individualteststores[features])
    test_probs = gbm.predict(dtest)
    indices = test_probs < 0
    test_probs[indices] = 0
    individualteststoreData = pd.DataFrame({'Id': individualteststores["Id"],
                                            'Sales': np.expm1(test_probs)  })
    if(teststoreData is None):
        teststoreData = individualteststoreData
    else:
        teststoreData = pd.concat((teststoreData,individualteststoreData))
    del dtest
    dtest = None
    gc.collect()
#trainstoreData = trainstoreData[['Store','Predictions','Sales']]
trainstoreData.to_csv('xgbtrainsubmission.csv',index=False)
teststoreData = teststoreData[['Id','Sales']]
teststoreData = teststoreData.sort('Id')
teststoreData.to_csv('xgbtestsubmission.csv',index=False)
print(totalerror/uniquestores.shape[0])
print('Finished')
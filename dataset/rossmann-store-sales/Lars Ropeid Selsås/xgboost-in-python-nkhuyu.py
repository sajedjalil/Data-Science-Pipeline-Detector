import pandas as pd
import numpy as np
import random
import math
import time
import xgboost as xgb

print(" -> Reading data files... ")
data_dir = '../input'
train = pd.read_csv('%s/train.csv' % data_dir)
test = pd.read_csv('%s/test.csv' % data_dir)
store = pd.read_csv('%s/store.csv' % data_dir)
store.fillna(-1, inplace=True)
print(" -> Remove columns with Sales = 0, Open = 0")
train = train[(train['Open']==1)]
print(" -> Join with Store table")
train = train.merge(store, on = 'Store', how = 'left')
test = test.merge(store, on = 'Store', how = 'left')
print(" -> Process the Date column")
for ds in [train, test]:
    tmpDate = [time.strptime(x, '%Y-%m-%d') for x in ds.Date]
    ds[  'mday'] = [x.tm_mday for x in tmpDate]
    ds[  'mon'] = [x.tm_mon for x in tmpDate]
    ds[  'year'] = [x.tm_year for x in tmpDate]
print(" -> Process categorical columns")
#for f in ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']:
for f in ['Store','DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'mday', 'mon', 'year']:
    tmp = train[f]
    tmp.append(test[f])
    tmp = list(tmp.unique())
    for ds in [train, test]:
        ds[f] = [tmp.index(x) for x in ds[f]]

store_features = ['StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
#features = ['Store','DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday'] 
#features = ['Store','DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'mday', 'mon', 'year'] 
features = ['DayOfWeek', 'Promo'] 
print(" -> XGBoost Train")
h = random.sample(range(len(train)),0)
dvalid = xgb.DMatrix(train.ix[h][features].values, label=[math.log(x+1) for x in train.ix[h]['Sales'].values])
dtrain = xgb.DMatrix(train.drop(h)[features].values, label=[x for x in train.drop(h)['Sales'].values])
param = {'objective': 'reg:linear', 
			  'eta': 1.0,
              'booster' : 'gbtree',
              'max_depth':10,
			  'subsample':1.0,
			  'silent' : 1,
			  'seed':100,
			  'colsample_bytree':1.0}
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    y = [x for x in labels[labels > 0]]
    yhat = [x for x in preds[labels > 0]]
    ssquare = [math.pow((y[i] - yhat[i])/y[i],2) for i in range(len(y))]
    return 'rmpse', math.sqrt(np.mean(ssquare))
watchlist = [(dtrain,'valid_rmpse')]
clf = xgb.train(param, dtrain, 1, watchlist,feval=evalerror)
dtest = xgb.DMatrix(test[test['Open']==1][features].values)
clf.dump_model('xgb.model.dump')

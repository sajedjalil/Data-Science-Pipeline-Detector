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
train = train[(train['Open']==1)&(train['Sales']>0)]
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
for f in ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']:
    tmp = train[f]
    tmp.append(test[f])
    tmp = list(tmp.unique())
    for ds in [train, test]:
        ds[f] = [tmp.index(x) for x in ds[f]]

store_features = ['StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
features = ['Store','DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'mday', 'mon', 'year'] + store_features
print(" -> XGBoost Train")
h = random.sample(range(len(train)),10000)
dvalid = xgb.DMatrix(train.ix[h][features].values, label=[math.log(x+1) for x in train.ix[h]['Sales'].values])
dtrain = xgb.DMatrix(train.drop(h)[features].values, label=[math.log(x+1) for x in train.drop(h)['Sales'].values])
param = {'objective': 'reg:linear', 
			  'eta': 0.05,
              'booster' : 'gbtree',
              'max_depth':10,
			  'subsample':0.9,
			  'silent' : 1,
			  'seed':20,
			  'colsample_bytree':0.7}
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    y = [math.exp(x)-1 for x in labels[labels > 0]]
    yhat = [math.exp(x)-1 for x in preds[labels > 0]]
    ssquare = [math.pow((y[i] - yhat[i])/y[i],2) for i in range(len(y))]
    return 'rmpse', math.sqrt(np.mean(ssquare))
watchlist = [(dvalid,'valid_rmpse')]
clf = xgb.train(param, dtrain, 2000, watchlist,feval=evalerror)
dtest = xgb.DMatrix(test[test['Open']==1][features].values)
test['Sales'] = 0
test.loc[test['Open']==1,'Sales'] = [math.exp(x) - 1 for x in clf.predict(dtest)] 
print("-> Write submission file ... ")
test[['Id', 'Sales']].to_csv("submission.csv", index = False)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from numpy import savetxt
from sklearn import preprocessing

train_file = '../input/train.csv'
test_file = '../input/test.csv'
store_file = '../input/store.csv'
output_file = 'predictions.csv'

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )
store = pd.read_csv(store_file)

train = pd.merge(train,store,on='Store')
test = pd.merge(test,store,on='Store')

train['Date'] = pd.to_datetime(train['Date'], coerce=True)
test['Date'] = pd.to_datetime(test['Date'], coerce=True)

train['year'] = pd.DatetimeIndex(train['Date']).year
train['month'] = pd.DatetimeIndex(train['Date']).month

test['year'] = pd.DatetimeIndex(test['Date']).year
test['month'] = pd.DatetimeIndex(test['Date']).month

train['logSale'] = np.log1p(train.Sales)


le = preprocessing.LabelEncoder()
le.fit(['a','b','c','d'])
train['labled_StoreType'] = le.transform(train.StoreType)
test['labled_StoreType'] = le.transform(test.StoreType)

le.fit(['a','b','c'])
train['labled_Assortment'] = le.transform(train.Assortment)
test['labled_Assortment'] = le.transform(test.Assortment)

train_holiday_in_str = train.StateHoliday.astype(str)
test_holiday_in_str = test.StateHoliday.astype(str)
le.fit(['0','a','b','c'])
train['labled_StateHoliday'] = le.transform(train_holiday_in_str)
test['labled_StateHoliday'] = le.transform(test_holiday_in_str)


#cols = [col for col in train.columns if col not in ['Id','Date','Sales','logSale','Customers', 'PromoInterval', 'StateHoliday', 'StoreType', 'Assortment']]
#features = train[cols]
#output = train['logSale']
#cols_pruned = [col for col in features.columns if col not in ['CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear']]
#features_pruned = train[cols_pruned]


#cols = [col for col in test.columns if col not in ['Id','Date','Sales','logSales','Customers', 'PromoInterval', 'StateHoliday', 'StoreType', 'Assortment']]
#test_features = test[cols]
#cols_pruned_test = [col for col in test_features.columns if col not in ['CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear']]
#test_features_pruned = test[cols_pruned_test]

output = train['logSale']
train.drop(['Date','Sales','logSale','Customers', 'PromoInterval', 'StateHoliday', 'StoreType', 'Assortment'], axis=1, inplace=True);
train.drop(['CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear'], axis=1 , inplace=True);

test.drop(['Id','Date', 'PromoInterval', 'StateHoliday', 'StoreType', 'Assortment'], axis=1, inplace = True);
test.drop(['CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear'], axis=1, inplace = True);
test.Open.fillna(1, inplace=True);

print(train.dtypes)
print(test.dtypes)

#rf = RandomForestClassifier(n_estimators=100, max_depth=30)
rf = RandomForestRegressor(n_estimators=120, max_depth=30)
rf.fit(train,output)

predicted_probs = [[index + 1, np.expm1(x)] for index, x in enumerate(rf.predict(test))]

savetxt('submission.csv', predicted_probs, delimiter=',', fmt='%d,%f', header='Id,Sales', comments = '')
     

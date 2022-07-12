import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import math

cal = calendar()
dr = pd.date_range(start='2016-01-01' ,end='2017-01-01')
holidays = cal.holidays(start=dr.min(), end=dr.max())

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

def preprocess(train):
    train.sort_values(by=['pickup_datetime'],inplace=True)
    train.loc[train.vendor_id==2,'vendor_id'] = 0
    train.loc[train.store_and_fwd_flag == 'N', 'store_and_fwd_flag'] = 1
    train.loc[train.store_and_fwd_flag == 'Y', 'store_and_fwd_flag'] = 0
    train.pickup_datetime = pd.DatetimeIndex(train.pickup_datetime)
    train['is_weekend']= train.pickup_datetime.apply(lambda x: int(x.day//5==1))
    train['hour_oftheday']= train.pickup_datetime.apply(lambda x: x.hour)
    train['minute_oftheday'] = train.pickup_datetime.apply(lambda x: x.minute)
    train['day_oftheweek'] = train.pickup_datetime.apply(lambda x: x.day % 7)
    train['is_holliday'] = train.pickup_datetime.apply(lambda x: x.date() in holidays)
    train['distance'] = ((train.dropoff_latitude - train.pickup_latitude)**2 +
                         (train.dropoff_longitude - train.pickup_longitude)**2)**0.5
    return train
    
train = preprocess(train)
features =['is_holliday','is_weekend', 'hour_oftheday', 'day_oftheweek','passenger_count', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude','vendor_id','passenger_count','minute_oftheday','distance']

X = train[features].values.astype(float)
y = train['trip_duration'].values.astype(float)
scl = StandardScaler()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.3)

X_train = scl.fit_transform(X_train)
X_test = scl.transform(X_test)
reg=xgb.XGBRegressor(max_depth=3,learning_rate=0.1)

reg.fit(X_train,y_train)
preds_train = reg.predict(X_train) 
preds_test = reg.predict(X_test)

preds_train[preds_train<0] = np.median(preds_train[preds_train>=0])
preds_test[preds_test<0] = np.median(preds_test[preds_test>=0])

print('train error:',rmsle(y_train,preds_train))
print('test error:',rmsle(y_test,preds_test))

test = preprocess(test)
X_sub = test[features].values.astype(float)
X_sub = scl.transform(X_sub)
y_sub = reg.predict(X_sub)
y_sub[y_sub<0] = np.median(y_sub[y_sub>=0])
test['trip_duration']=y_sub
sub = test[['id','trip_duration']]
sub.to_csv('submission.csv',index=False)


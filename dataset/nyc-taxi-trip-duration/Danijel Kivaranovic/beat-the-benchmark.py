
## libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

print('Read data...')
## read data
train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')
test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')

## dates
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])

## transform character to numeric
le = LabelEncoder()
le.fit(train['store_and_fwd_flag'])
train['store_and_fwd_flag'] = le.transform(train['store_and_fwd_flag'])
test['store_and_fwd_flag'] = le.transform(test['store_and_fwd_flag'])

###############################################################################
## New features
###############################################################################
print('Create features...')
#### date features
train['month'] = train['pickup_datetime'].dt.month
train['day'] = train['pickup_datetime'].dt.day
train['weekday'] = train['pickup_datetime'].dt.weekday
train['hour'] = train['pickup_datetime'].dt.hour
train['minute'] = train['pickup_datetime'].dt.minute

test['month'] = test['pickup_datetime'].dt.month
test['day'] = test['pickup_datetime'].dt.day
test['weekday'] = test['pickup_datetime'].dt.weekday
test['hour'] = test['pickup_datetime'].dt.hour
test['minute'] = test['pickup_datetime'].dt.minute

#### distance features
train['dist_long'] = train['pickup_longitude'] - train['dropoff_longitude']
test['dist_long'] = test['pickup_longitude'] - test['dropoff_longitude']

train['dist_lat'] = train['pickup_latitude'] - train['dropoff_latitude']
test['dist_lat'] = test['pickup_latitude'] - test['dropoff_latitude']

train['dist'] = np.sqrt(np.square(train['dist_long']) + np.square(train['dist_lat']))
test['dist'] = np.sqrt(np.square(test['dist_long']) + np.square(test['dist_lat']))

#### spatial features: count and speed
train['pickup_longitude_bin'] = np.round(train['pickup_longitude'], 2)
train['pickup_latitude_bin'] = np.round(train['pickup_latitude'], 2)
train['dropoff_longitude_bin'] = np.round(train['dropoff_longitude'], 2)
train['dropoff_latitude_bin'] = np.round(train['dropoff_latitude'], 2)

test['pickup_longitude_bin'] = np.round(test['pickup_longitude'], 2)
test['pickup_latitude_bin'] = np.round(test['pickup_latitude'], 2)
test['dropoff_longitude_bin'] = np.round(test['dropoff_longitude'], 2)
test['dropoff_latitude_bin'] = np.round(test['dropoff_latitude'], 2)

## count features
a = pd.concat([train,test]).groupby(['pickup_longitude_bin', 'pickup_latitude_bin']).size().reset_index()
b = pd.concat([train,test]).groupby(['dropoff_longitude_bin', 'dropoff_latitude_bin']).size().reset_index()

train = pd.merge(train, a, on = ['pickup_longitude_bin', 'pickup_latitude_bin'], how = 'left')
test = pd.merge(test, a, on = ['pickup_longitude_bin', 'pickup_latitude_bin'], how = 'left')

train = pd.merge(train, b, on = ['dropoff_longitude_bin', 'dropoff_latitude_bin'], how = 'left')
test = pd.merge(test, b, on = ['dropoff_longitude_bin', 'dropoff_latitude_bin'], how = 'left')

## speed features
train['speed'] = 100000*train['dist'] / train['trip_duration']

a = train[['speed', 'pickup_longitude_bin', 'pickup_latitude_bin']].groupby(['pickup_longitude_bin', 'pickup_latitude_bin']).mean().reset_index()
a = a.rename(columns = {'speed': 'ave_speed'})
b = train[['speed', 'dropoff_longitude_bin', 'dropoff_latitude_bin']].groupby(['dropoff_longitude_bin', 'dropoff_latitude_bin']).mean().reset_index()
b = b.rename(columns = {'speed': 'ave_speed'})

train = pd.merge(train, a, on = ['pickup_longitude_bin', 'pickup_latitude_bin'], how = 'left')
test = pd.merge(test, a, on = ['pickup_longitude_bin', 'pickup_latitude_bin'], how = 'left')

train = pd.merge(train, b, on = ['dropoff_longitude_bin', 'dropoff_latitude_bin'], how = 'left')
test = pd.merge(test, b, on = ['dropoff_longitude_bin', 'dropoff_latitude_bin'], how = 'left')

## drop bins
train = train.drop(['speed', 'pickup_longitude_bin', 'pickup_latitude_bin', 'dropoff_longitude_bin', 'dropoff_latitude_bin'], axis = 1)
test = test.drop(['pickup_longitude_bin', 'pickup_latitude_bin', 'dropoff_longitude_bin', 'dropoff_latitude_bin'], axis = 1)

#### weather data
weather = pd.read_csv('../input/knycmetars2016/KNYC_Metars.csv')
weather['Time'] = pd.to_datetime(weather['Time'])
weather['year'] = weather['Time'].dt.year
weather['month'] = weather['Time'].dt.month
weather['day'] = weather['Time'].dt.day
weather['hour'] = weather['Time'].dt.hour
weather = weather[weather['year'] == 2016]

train = pd.merge(train, weather[['Temp.', 'month', 'day', 'hour']], on = ['month', 'day', 'hour'], how = 'left')
test = pd.merge(test, weather[['Temp.', 'month', 'day', 'hour']], on = ['month', 'day', 'hour'], how = 'left')

## train/test features, y, id
xtrain = train.drop(['id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration'], axis = 1).as_matrix()
xtest = test.drop(['id', 'pickup_datetime', ], axis = 1).as_matrix()
ytrain = train['trip_duration'].values
id_train = train['id'].values
id_test = test['id'].values
del(train, test)

## xgb parameters
params = {
    'booster':            'gbtree',
    'objective':          'reg:linear',
    'learning_rate':      0.1,
    'max_depth':          14,
    'subsample':          0.8,
    'colsample_bytree':   0.7,
    'colsample_bylevel':  0.7,
    'silent':             1
}

## number of rounds
nrounds = 200

## train model
print('Train model...')
dtrain = xgb.DMatrix(xtrain, np.log(ytrain+1))
gbm = xgb.train(params,
                dtrain,
                num_boost_round = nrounds)

## test predictions
pred_test = np.exp(gbm.predict(xgb.DMatrix(xtest))) - 1

## create submission
df = pd.DataFrame({'id': id_test, 'trip_duration': pred_test}) 
df = df.set_index('id')
df.to_csv('sub_bench.csv', index = True)






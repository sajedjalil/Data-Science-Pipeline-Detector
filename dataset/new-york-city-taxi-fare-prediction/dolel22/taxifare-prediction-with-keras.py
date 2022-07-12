# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

## Script is a combination of cleaning techniques I picked up from other kernels and applied in this before running my own version of a keras MLP

import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# show tensorflow version
print(tf.__version__)

# options for models
pd.set_option('display.width', 900)

# start working on model
df = (pd
      .read_csv('../input/train.csv', nrows=5000000, encoding='utf8')
      .drop(['key'], axis=1))

df.pickup_datetime = pd.to_datetime(df.pickup_datetime, format='%Y-%m-%d %H:%M:%S %Z')
df.pickup_latitude = df.pickup_latitude.astype('float32')
df.pickup_longitude = df.pickup_longitude.astype('float32')
df.dropoff_latitude = df.dropoff_latitude.astype('float32')
df.dropoff_longitude = df.dropoff_longitude.astype('float32')
df.fare_amount = df.fare_amount.astype('float32')
df.passenger_count = df.passenger_count.astype('uint8')
print(df.shape)

# clean up data a little
df = df[(-76 <= df['pickup_longitude']) & (df['pickup_longitude'] <= -72)]
df = df[(-76 <= df['dropoff_longitude']) & (df['dropoff_longitude'] <= -72)]
df = df[(38 <= df['pickup_latitude']) & (df['pickup_latitude'] <= 42)]
df = df[(38 <= df['dropoff_latitude']) & (df['dropoff_latitude'] <= 42)]
df = df[(1 <= df['passenger_count']) & (df['passenger_count'] <= 6)]
df = df[df['fare_amount'] > 0]
print(df.shape)

# compute distances
def degree_to_radion(degree):
    return degree * (np.pi / 180)

def calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
    from_lat = degree_to_radion(pickup_latitude)
    from_long = degree_to_radion(pickup_longitude)
    to_lat = degree_to_radion(dropoff_latitude)
    to_long = degree_to_radion(dropoff_longitude)
    
    radius = 6371.01
    
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = (np.sin(lat_diff / 2) ** 2 
         + np.cos(degree_to_radion(from_lat)) 
         * np.cos(degree_to_radion(to_lat)) 
         * np.sin(long_diff / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c

def add_features(data):
    data['hour_of_day'] = data.pickup_datetime.dt.hour.astype('category')
    data['day_of_week'] = data.pickup_datetime.dt.dayofweek.astype('category')
    data['day_of_month'] = data.pickup_datetime.dt.day.astype('category')
    data['week_of_year'] = data.pickup_datetime.dt.weekofyear.astype('category')
    data['month_of_year'] = data.pickup_datetime.dt.month.astype('category')
    data['quarter_of_year'] = data.pickup_datetime.dt.quarter.astype('category')
    data['year'] = data.pickup_datetime.dt.year.astype('category')
    
    lgr = (-73.8733, 40.7746)
    jfk = (-73.7900, 40.6437)
    ewr = (-74.1843, 40.6924)

    data['trip_distance_km'] = calculate_distance(data.pickup_latitude, data.pickup_longitude, data.dropoff_latitude, data.dropoff_longitude).astype('float32')
    data['pickup_distance_jfk'] = calculate_distance(data['pickup_latitude'], data['pickup_longitude'], jfk[1], jfk[0]).astype('float32')
    data['dropoff_distance_jfk'] = calculate_distance(data['dropoff_latitude'], data['dropoff_longitude'], jfk[1], jfk[0]).astype('float32')
    data['pickup_distance_ewr'] = calculate_distance(data['pickup_latitude'], data['pickup_longitude'], ewr[1], ewr[0]).astype('float32')
    data['dropoff_distance_ewr'] = calculate_distance(data['dropoff_latitude'], data['dropoff_longitude'], ewr[1], ewr[0]).astype('float32')
    data['pickup_distance_laguardia'] = calculate_distance(data['pickup_latitude'], data['pickup_longitude'], lgr[1], lgr[0]).astype('float32')
    data['dropoff_distance_laguardia'] = calculate_distance(data['dropoff_latitude'], data['dropoff_longitude'], lgr[1], lgr[0]).astype('float32')    
    return data

df = add_features(df)
print(df.head())
    
from sklearn.preprocessing import StandardScaler
numerics = [
    'passenger_count',                         
    'trip_distance_km',
    'pickup_distance_jfk',
    'dropoff_distance_jfk',
    'pickup_distance_ewr',
    'dropoff_distance_ewr',
    'pickup_distance_laguardia',
    'dropoff_distance_laguardia' 
]
standard_scaler = StandardScaler().fit(df[numerics])
numeric_scale = standard_scaler.transform(df[numerics])

from sklearn.preprocessing import OneHotEncoder

cat_cols = [
    'hour_of_day',
    'day_of_week',
    'day_of_month',
    'week_of_year',
    'month_of_year',
    'quarter_of_year',
    'year'
]
cat_encoder = OneHotEncoder(sparse=False).fit(df[cat_cols])
categorical_scale = cat_encoder.transform(df[cat_cols])


X = np.hstack((categorical_scale, numeric_scale))
del categorical_scale
del numeric_scale


def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(X.shape[1], activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(1, activation='linear'))

model.compile(
    loss='mean_squared_error', 
    optimizer='adam',
    metrics=[rmse, tf.keras.losses.mean_absolute_error]
)

y = df.fare_amount.values
history = model.fit(
    x=X, 
    y=y,
    validation_split=0.1,
    epochs=15,
    batch_size=500, 
    shuffle=True, 
    verbose=2
)


print('generating predictions for test data')
test = pd.read_csv('../input/test.csv')
test.pickup_datetime = pd.to_datetime(test.pickup_datetime, format='%Y-%m-%d %H:%M:%S %Z')
print(test.shape)

print('adding features to test data')
test = add_features(test)

print('calling model for predictions')
predictions = model.predict(
    np.hstack((cat_encoder.transform(test[cat_cols]), standard_scaler.transform(test[numerics])))
)
submission = pd.DataFrame(predictions, columns=['fare_amount'])
submission['key'] = test['key']

submission[['key', 'fare_amount']].to_csv('code_kernel_submission.csv', index=False)
print(submission.head())






















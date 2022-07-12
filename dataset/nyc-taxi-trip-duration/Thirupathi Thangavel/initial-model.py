import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from haversine import haversine

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Train on lower number of points if running on laptop
#train = train.head(30000)

# Remove passenger count outliers
train = train[train['passenger_count'] > 0]
train = train[train['passenger_count'] < 9]

# Remove coordinate outliers
train = train[train['pickup_longitude'] <= -73.75]
train = train[train['pickup_longitude'] >= -74.03]
train = train[train['pickup_latitude'] <= 40.85]
train = train[train['pickup_latitude'] >= 40.63]
train = train[train['dropoff_longitude'] <= -73.75]
train = train[train['dropoff_longitude'] >= -74.03]
train = train[train['dropoff_latitude'] <= 40.85]
train = train[train['dropoff_latitude'] >= 40.63]

# Remove trip_duration outliers
trip_duration_mean = np.mean(train['trip_duration'])
trip_duration_std = np.std(train['trip_duration'])
train = train[train['trip_duration'] <= trip_duration_mean + 2 * trip_duration_std]
train = train[train['trip_duration'] >= trip_duration_mean - 2 * trip_duration_std]

# Convert to datetime object to easily extract hour of day, day of week
train['pickup_datetime_object'] = pd.to_datetime(train['pickup_datetime'])
test['pickup_datetime_object'] = pd.to_datetime(test['pickup_datetime'])

# Day of week and hour of day affects the ride duration most
train['day_of_week'] = train['pickup_datetime_object'].apply(lambda x: x.weekday())
test['day_of_week'] = test['pickup_datetime_object'].apply(lambda x: x.weekday())
train['hour_of_day'] = train['pickup_datetime_object'].apply(lambda x: x.hour)
test['hour_of_day'] = test['pickup_datetime_object'].apply(lambda x: x.hour)

# Convert day of week and hour of day to cyclic encoding
# Otherwise day 0 (Sunday) and day 6 (Saturday) are far apart, which is wrong
def day_of_week_sine(x):
    return np.sin(2 * np.pi * x / 7)

train['day_of_week_sine'] = train['day_of_week'].apply(day_of_week_sine)
test['day_of_week_sine'] = test['day_of_week'].apply(day_of_week_sine)

def hour_of_day_sine(x):
    return np.sin(2 * np.pi * x / 24)

train['hour_of_day_sine'] = train['hour_of_day'].apply(hour_of_day_sine)
test['hour_of_day_sine'] = test['hour_of_day'].apply(hour_of_day_sine)

# Compute the aerial distance. 
# Not considering the curvature of earth, 
# since all data points are localized to a very small area - New York city
train['dist_long'] = train['pickup_longitude'] - train['dropoff_longitude']
test['dist_long'] = test['pickup_longitude'] - test['dropoff_longitude']
train['dist_lat'] = train['pickup_latitude'] - train['dropoff_latitude']
test['dist_lat'] = test['pickup_latitude'] - test['dropoff_latitude']

train['distance'] = np.sqrt(np.square(train['dist_long']) + np.square(train['dist_lat']))
test['distance'] = np.sqrt(np.square(test['dist_long']) + np.square(test['dist_lat']))
train['distance_haver'] = train.apply(lambda x: haversine((x["pickup_longitude"], x["pickup_latitude"]), (x["dropoff_longitude"], x["dropoff_latitude"])), axis=1)
test['distance_haver'] = test.apply(lambda x: haversine((x["pickup_longitude"], x["pickup_latitude"]), (x["dropoff_longitude"], x["dropoff_latitude"])), axis=1)

# View the correlation between each other variables
print(train.corr())

# Select only few variables as input, based on correlation analysis
train_x = train[['passenger_count', 'vendor_id', 'distance', 'day_of_week_sine', 'hour_of_day_sine']]
#train_x = train[['passenger_count', 'vendor_id', 'distance']] # Remove temporarily to reduce number of variables
train_y = train['trip_duration']
test_x = test[['passenger_count', 'vendor_id', 'distance', 'day_of_week_sine', 'hour_of_day_sine']]
#test_x = test[['passenger_count', 'vendor_id', 'distance']] # Remove temporarily to reduce number of variables

# Since number of data points is huge, 
# use linear regression model with low tolerance, and saga solver
# otherwise it takes to months to train. 
model = LogisticRegression(tol=0.1, solver='saga')
model = LinearRegression()
model.fit(train_x, train_y)

test['trip_duration'] = model.predict(test_x)

# We need to submit only the id and predicted trip duration
test[['id', 'trip_duration']].to_csv('submission.csv', index=False)


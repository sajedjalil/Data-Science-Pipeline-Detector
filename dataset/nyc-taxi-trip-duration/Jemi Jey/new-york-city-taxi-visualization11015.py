import os
from datetime import timedelta

# import numpy as np
# import pandas as pd
# import haversine

# train_data_frame = pd.read_csv('../input/train.csv')
# test_data_frame = pd.read_csv('../input/test.csv')

# train_data = pd.read_csv('../input/train1/train1.csv')

# #Converting Trip duration in Hours
# train_data['trip_dur_hr']=train_data['trip_duration']/30
# print(train_data['trip_dur_hr'].describe(),'\n')

# print(train_data.trip_duration.describe())

# train_data['distance'] = train_data.apply(lambda x: haversine.haversine((x["pickup_longitude"], x["pickup_latitude"]), (x["dropoff_longitude"], x["dropoff_latitude"])), axis=1)

# print(train_data.head())
# ----------------------------------------------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

import haversine

print(os.listdir())


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


train_data['pickup_datetime'] = pd.to_datetime(train_data['pickup_datetime'])
train_data['dropoff_datetime'] = pd.to_datetime(train_data['dropoff_datetime'])

train_data['pickup_month'] = train_data.pickup_datetime.dt.month.astype(np.uint8)
train_data['pickup_day'] = train_data.pickup_datetime.dt.weekday.astype(np.uint8)
train_data['pickup_hour'] = train_data.pickup_datetime.dt.hour.astype(np.uint8)

train_data['dropoff_month'] = train_data.dropoff_datetime.dt.month.astype(np.uint8)
train_data['dropoff_day'] = train_data.dropoff_datetime.dt.weekday.astype(np.uint8)
train_data['dropoff_hour'] = train_data.dropoff_datetime.dt.hour.astype(np.uint8)

train_data['distance'] = train_data.apply(lambda x: haversine.haversine((x["pickup_longitude"], x["pickup_latitude"]), (x["dropoff_longitude"], x["dropoff_latitude"])), axis=1)

# --- Doing the same for the test data excluding dropoff time ---
test_data['pickup_datetime'] = pd.to_datetime(test_data['pickup_datetime'])

test_data['pickup_month'] = test_data.pickup_datetime.dt.month.astype(np.uint8)
test_data['pickup_day'] = test_data.pickup_datetime.dt.weekday.astype(np.uint8)
test_data['pickup_hour'] = test_data.pickup_datetime.dt.hour.astype(np.uint8)

train_data['trip_duration_mins'] = train_data['trip_duration'] / 60
train_data['trip_duration_hours'] = train_data['trip_duration_mins'] / 60

print(max(train_data['trip_duration_hours']))
print(min(train_data['trip_duration_hours']))

print(train_data['trip_duration_hours'].describe())

print(train_data[train_data['trip_duration_hours'] > 5].count()['id'])
print(len(train_data))

train_data.drop(train_data[train_data.trip_duration_hours > 5].index, inplace=True)
print(len(train_data))

print(max(train_data['trip_duration_hours']))
print(min(train_data['trip_duration_hours']))

print(train_data['trip_duration_hours'].describe())

print(train_data['passenger_count'].unique())
print(test_data['passenger_count'].unique())
print(test_data[(test_data['passenger_count'] == 0)].count()['id'])

plt.plot(train_data['pickup_longitude'], train_data['pickup_latitude'], '.', color='k', alpha=0.8)
plt.title('Pickup Location Lat and Long', weight='bold')
plt.show()

plt.plot(train_data['dropoff_longitude'], train_data['dropoff_latitude'], '.', color='k', alpha=0.8)
plt.title('Dropoff Location Lat and Long', weight='bold')
plt.show()

# Remove latitude and longtitude outlier
train_data = train_data[train_data.pickup_latitude != 51.881084442138672]
train_data = train_data[train_data.pickup_longitude != -121.93334197998048]
train_data = train_data[train_data.dropoff_longitude != -121.93320465087892]
train_data = train_data[train_data.dropoff_latitude != 32.181140899658203]

plt.plot(train_data['pickup_longitude'], train_data['pickup_latitude'], '.', color='k', alpha=0.8)
plt.title('Pickup Location Lat and Long', weight = 'bold')
plt.show()

plt.plot(train_data['dropoff_longitude'], train_data['dropoff_latitude'], '.', color='k', alpha=0.8)
plt.title('Dropoff Location Lat and Long', weight = 'bold')
plt.show()

mean_pickup_lat = np.mean(train_data['pickup_latitude'])
mean_pickup_lon = np.mean(train_data['pickup_longitude'])

print(mean_pickup_lat)
print(mean_pickup_lon)

# Standard deviation of pickup & dropoff Lats and Longs
std_pickup_lat = np.std(train_data['pickup_latitude'])
std_pickup_lon = np.std(train_data['pickup_longitude'])

print(std_pickup_lat)
print(std_pickup_lon)

min_pickup_lat = mean_pickup_lat - std_pickup_lat
max_pickup_lat = mean_pickup_lat + std_pickup_lat
min_pickup_lon = mean_pickup_lon - std_pickup_lon
max_pickup_lon = mean_pickup_lon + std_pickup_lon

locations = train_data[(train_data.pickup_latitude > min_pickup_lat) &
                       (train_data.pickup_latitude < max_pickup_lat) &
                       (train_data.pickup_longitude > min_pickup_lon) &
                       (train_data.pickup_longitude < max_pickup_lon)]

plt.plot(locations['pickup_longitude'], locations['pickup_latitude'], '.', color='k', alpha=0.8)
plt.title('Reduced Pickup Lat and Long', weight='bold')
plt.show()

print(len(train_data))
print(len(locations))

# making a duplicate copy of the df to work on
locations_1 = locations
print(locations_1.head())

# Assigning the target variable
labels = train_data['trip_duration']

print(train_data.head())

# convert the categorical variables to numerical variables
df_s_f_train = pd.get_dummies(train_data['store_and_fwd_flag'])
df_s_f_test = pd.get_dummies(test_data['store_and_fwd_flag'])

# Join the dummy variables to the main dataframe
train_data = pd.concat([train_data, df_s_f_train], axis=1)
test_data = pd.concat([test_data, df_s_f_test], axis=1)

# Drop the categorical column
train_data.drop('store_and_fwd_flag', axis=1, inplace=True)
test_data.drop('store_and_fwd_flag', axis=1, inplace=True)

train_data = train_data.loc[:, ~train_data.columns.duplicated()]
test_data = test_data.loc[:, ~test_data.columns.duplicated()]

train_data.drop('id', axis=1, inplace=True)
print(train_data.isnull().values.any())
print(test_data.isnull().values.any())

print(train_data.head())
print(test_data.head())

b_train = train_data.drop(['pickup_datetime', 'dropoff_datetime',
                           'dropoff_hour', 'dropoff_month',
                           'dropoff_day', 'trip_duration',
                           'trip_duration_mins', 'trip_duration_hours'], 1)
b_label = train_data['trip_duration']

test = test_data
test = test.drop(['pickup_datetime', 'id'], 1)
print(test.head())

# List of important features
RF = RandomForestRegressor()
RF.fit(b_train, b_label)

print(RF)

features_list = b_train.columns.values
feature_importance = RF.feature_importances_
sorted_idx = np.argsort(feature_importance)

print(sorted_idx)

plt.figure(figsize=(15, 15))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
plt.show()

min_pickup_lat = mean_pickup_lat - (3 * std_pickup_lat)
max_pickup_lat = mean_pickup_lat + (3 * std_pickup_lat)
min_pickup_lon = mean_pickup_lon - (3 * std_pickup_lon)
max_pickup_lon = mean_pickup_lon + (3 * std_pickup_lon)

# min_pickup_lat = mean_pickup_lat - (4 * std_pickup_lat)
# max_pickup_lat = mean_pickup_lat + (4 * std_pickup_lat)
# min_pickup_lon = mean_pickup_lon - (4 * std_pickup_lon)
# max_pickup_lon = mean_pickup_lon + (4 * std_pickup_lon)
#
# min_pickup_lat = mean_pickup_lat - (10 * std_pickup_lat)
# max_pickup_lat = mean_pickup_lat + (10 * std_pickup_lat)
# min_pickup_lon = mean_pickup_lon - (10 * std_pickup_lon)
# max_pickup_lon = mean_pickup_lon + (10 * std_pickup_lon)
#
# min_pickup_lat = mean_pickup_lat - (5 * std_pickup_lat)
# max_pickup_lat = mean_pickup_lat + (5 * std_pickup_lat)
# min_pickup_lon = mean_pickup_lon - (5 * std_pickup_lon)
# max_pickup_lon = mean_pickup_lon + (5 * std_pickup_lon)

locations = train_data[(train_data.pickup_latitude > min_pickup_lat) &
                       (train_data.pickup_latitude < max_pickup_lat) &
                       (train_data.pickup_longitude > min_pickup_lon) &
                       (train_data.pickup_longitude < max_pickup_lon)]

plt.plot(locations['pickup_longitude'], locations['pickup_latitude'], '.', color='k', alpha=0.8)
plt.title('Reduced Pickup Lat and Long', weight = 'bold')
plt.show()

# should remove till this
# train_data.drop('id', axis=1, inplace=True)



















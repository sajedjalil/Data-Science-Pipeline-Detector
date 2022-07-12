# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import dask.dataframe as dd
import os
from tqdm import tqdm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Set columns to most suitable type to optimize for memory usage
"""TRAIN_PATH = '../input/train.csv'
traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

cols = list(traintypes.keys())

chunksize = 5_000_000 # 5 million rows at one go. Or try 10 million


df_list = [] # list to hold the batch dataframe

for df_chunk in tqdm(pd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes, chunksize=chunksize)):
     
    # Neat trick from https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
    # Using parse_dates would be much slower!
    df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)
    df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    
    # Can process each chunk of dataframe here
    # clean_data(), feature_engineer(),fit()
    
    # Alternatively, append the chunk to list and merge all
    df_list.append(df_chunk) 

train = pd.concat(df_list)

# Delete the dataframe list to release memory
del df_list
"""

train=pd.read_csv('../input/train.csv',nrows=1000000)
test = pd.read_csv('../input/test.csv')
train = train.dropna()
train = train.loc[~(train==0).any(axis =1)]

object_data = train.dtypes == np.object
categoricals = train.columns[object_data]

train.drop('key', axis = 1, inplace = True)
import datetime as dt

def date_extraction(data):
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    data['year'] = data['pickup_datetime'].dt.year
    data['month'] = data['pickup_datetime'].dt.month
    data['weekday'] = data['pickup_datetime'].dt.day
    data['hour'] = data['pickup_datetime'].dt.hour
    data = data.drop('pickup_datetime', axis = 1, inplace = True)
    
    return data
    
#Apply this to both the train and the test data
date_extraction(train)
date_extraction(test)

def long_lat_distance (x):
    x['Longitude_distance'] = np.radians(x['pickup_longitude'] - x['dropoff_longitude'])
    x['Latitude_distance'] = np.radians(x['pickup_latitude'] - x['dropoff_latitude']) 
    x['distance_travelled/10e3'] = ((x['Longitude_distance']**2 + x['Latitude_distance']**2)**0.5) *1000
    return x   
    
for x in [train, test]:
    long_lat_distance(x)
    
def harvesine(x):
    #radii of earth in meters = 
    r = 6371000 
    d = x['distance_travelled/10e3']
    theta_1 = np.radians(x['dropoff_latitude'])
    theta_2 = np.radians(x['pickup_latitude'])
    lambda_1 = np.radians(x['dropoff_longitude'])
    lambda_2 = np.radians(x['dropoff_longitude'])
    theta_diff = x['Longitude_distance']
    lambda_diff = x['Latitude_distance']
    
    a = np.sin(theta_diff/2)**2 + np.cos(theta_1)*np.cos(theta_2)*np.sin(lambda_diff/2)**2
    c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
    x['harvesine/km'] = (r * c)/1000
    
for x in [train, test]:
    harvesine(x)
train['harvesine/km'] = train['harvesine/km'].fillna(train['harvesine/km'].median())


from sklearn.ensemble import RandomForestRegressor

#split the train features 
feature_cols = [x for x in train.columns if x!= 'fare_amount']
X = train[feature_cols]
y = train['fare_amount']

correlations = X.corrwith(y)
correlations = abs(correlations*100)
correlations.sort_values(ascending = False, inplace= True)

train_1 = train.drop(['pickup_longitude', 'dropoff_longitude','pickup_latitude','dropoff_latitude',
                    'Longitude_distance', 'Latitude_distance'], axis =1)
train_1['harvesine/km'] =train_1['harvesine/km'].round(2) 
train_1['distance_travelled/10e3'] =train_1['distance_travelled/10e3'].round(2) 

test_1 = test.drop(['pickup_longitude', 'dropoff_longitude','pickup_latitude','dropoff_latitude',
                    'Longitude_distance', 'Latitude_distance'], axis =1)

from sklearn.model_selection import train_test_split
feat_cols = [x for x in train_1.columns if x!= 'fare_amount']
X_1 = train_1[feat_cols]
y_1 = train_1['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size = 0.25, random_state = 42)

#Random forest
rf = RandomForestRegressor(n_estimators = 100, max_features = 5)
rf = rf.fit(X_train, y_train)

final_prediction = rf.predict(X_test)

test_1.drop('key', axis = 1, inplace = True)

#random forest
final_prediction = rf.predict(test_1)

NYCtaxiFare_submission = pd.DataFrame({'key': test.key, 'fare_amount': final_prediction})
NYCtaxiFare_submission.to_csv('NYCtaxiFare_prediction.csv', index = False)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Good starting point if you'd like to experiment with Keras
# I have achieved ~.491x with various adjustments, but it
# takes a while to run

import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras import optimizers
from keras.callbacks import EarlyStopping

# Data wrangling brought to you by the1owl
# https://www.kaggle.com/the1owl/surprise-me

data = {
    'tra': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'tes': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_visitors':'rv1'})
    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_visitors':'rv2'})
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

#sure it can be compressed...
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 
lbl = preprocessing.LabelEncoder()
#stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
#stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

# Georgii Vyshnia
# context: https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/discussion/44842
stores['prefecture'] = stores['air_area_name'].map(lambda x: str(x).split(' ')[0])

# Try OHE instead of LE
# CV did not improve
stores = pd.get_dummies(stores, prefix_sep='_', columns=['air_area_name'])
stores = pd.get_dummies(stores, prefix_sep='_', columns=['air_genre_name'])
stores = pd.get_dummies(stores, prefix_sep='_', columns=['prefecture'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

data['tra'] = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])
data['tes'] = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date'])

train = pd.merge(data['tra'], stores, how='left', on=['air_store_id','dow']) 
test = pd.merge(data['tes'], stores, how='left', on=['air_store_id','dow'])

for df in ['ar','hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 
    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])

train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
    
# Full date to int
# CV improved
train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

# Geo
# CV improved
train['var_max_lat'] = train['latitude'].max() - train['latitude']
train['var_max_long'] = train['longitude'].max() - train['longitude']
test['var_max_lat'] = test['latitude'].max() - test['latitude']
test['var_max_long'] = test['longitude'].max() - test['longitude']

# Georgii Vyshnia
train['lon_plus_lat'] = train['longitude'] + train['latitude']
test['lon_plus_lat'] = test['longitude'] + test['latitude']


train = train.fillna(-1)
test = test.fillna(-1)

x_train = train.drop(['air_store_id','visit_date','visitors'], axis=1)
y_train = np.log1p(train['visitors'].values)

y_test = test['visitors'].as_matrix()
x_test = test.drop(['id','air_store_id','visit_date','visitors'], axis=1)

x_train = x_train.as_matrix()
x_test = x_test.as_matrix()
y_train = pd.DataFrame(y_train).as_matrix()

# Define the scaler 
scaler = StandardScaler().fit(x_train)

# Scale the train set
x_train = scaler.transform(x_train)

# Scale the test set
x_test = scaler.transform(x_test)
    
# Set random seed
np.random.seed(7)

# Start neural network
network = models.Sequential()

# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=24, activation='relu', input_shape=(x_train.shape[1],)))

network.add(layers.Dropout(.1))

network.add(layers.Dense(units=20, activation='relu'))

network.add(layers.Dropout(.05))

network.add(layers.Dense(units=1))

# Optimizers
# TRY SGD + NESTEROV for shallow nets?
opt = optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
#opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Compile neural network
network.compile(loss='mse', # Mean squared error
            optimizer=opt, # Optimization algorithm
            metrics=['mse']) # Mean squared error

# Since we log transformed the target, the sq root of reported MSE = RMSLE

# Train neural network
network.fit(x_train, # Features
            y_train, # Target vector
            #callbacks=[early_stopping],
            epochs=200, # Number of epochs
            verbose=0, # Output
            batch_size=64 # Number of observations per batch
            )

keras_pred = network.predict(x_test)

test['visitors'] = np.absolute(np.expm1(keras_pred))

test[['id','visitors']].to_csv('keras_submission.csv', index=False, float_format='%.3f')
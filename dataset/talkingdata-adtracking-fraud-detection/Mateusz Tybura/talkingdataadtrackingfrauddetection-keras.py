import numpy as np # linear algebra
seed = 7
PYTHONHASHSEED = seed
np.random.seed(seed)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense, Dropout
import gc
import os
import time
from sklearn import preprocessing

os.environ['OMP_NUM_THREADS'] = '4'  # Number of threads on the Kaggle server

def freq_hours(row):
    if row['hour'] in [4, 5, 9, 10, 13, 14]:
        return 1 #most frequent hours in test data
    elif row['hour'] in [6, 11, 15]:
        return 2 #least frequent hours in test data
    return 3 #none of them

def features(df):
    print('Making features')
    
    print('1 - datetime')
    df['datetime'] = pd.to_datetime(df['click_time'])
    df['dow'] = df['datetime'].dt.dayofweek.astype('uint8')
    df['month'] = df['datetime'].dt.month.astype('uint8')
    df['day'] = df["datetime"].dt.day.astype('uint8')
    df['hour'] = df["datetime"].dt.hour.astype('uint8')
    df['freq_hour'] = df.apply(lambda row: freq_hours(row) , axis=1)
    df['is_am'] = df.apply(lambda row: row['hour'] <= 12, axis=1)
    df['on_weekend'] = df.apply(lambda row: row['dow'] >= 5, axis=1)
    df.drop(['click_time', 'datetime'], axis=1, inplace=True)
    gc.collect()
    
    print('2 - Number of clicks for ip')
    ip_clicks = df[['ip','channel']].groupby(by=['ip'])[['channel']]\
        .count().reset_index().rename(columns={'channel': 'n_ip_clicks'})
    df = df.merge(ip_clicks, on=['ip'], how='left')
    del ip_clicks
    gc.collect()
    
    print('3 - Number of channels for ip within hour')
    n_chans = df[['ip','day','hour','channel']].groupby(by=['ip','day',
              'hour'])[['channel']].count().reset_index().rename(columns={'channel': 'n_channels'})
    df = df.merge(n_chans, on=['ip','day','hour'], how='left')
    del n_chans
    gc.collect()

    print('4 - Number of channels for ip and app')
    n_chans = df[['ip','app', 'channel']].groupby(by=['ip',
              'app'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_count'})
    df = df.merge(n_chans, on=['ip','app'], how='left')
    del n_chans
    gc.collect()

    print('5 - Number of channels for ip, app and os')
    n_chans = df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app',
              'os'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_os_count'})
    df = df.merge(n_chans, on=['ip','app', 'os'], how='left')
    del n_chans
    gc.collect()

    print('Fixing types')
    df.info()
    for feat in ['n_channels', 'ip_app_count', 'ip_app_os_count', 'n_ip_clicks']:
        df[feat] = df[feat].astype('uint16')
        
    df.info()
    
    return df
    
def normalize(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    return pd.DataFrame(np_scaled)
    

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols  = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
col_types = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

#n_rows_to_skip = 110000000
rows_to_read = 3000000 #memory limits :(
print('Loading training data')
train_raw = pd.read_csv('../input/train.csv', usecols = train_cols, dtype=col_types, nrows = rows_to_read)
print('Training data loaded')
gc.collect()

print('Processing training data')
y = train_raw['is_attributed']
train_raw.drop(['is_attributed'], axis=1, inplace=True)
gc.collect()
featured = features(train_raw)
del train_raw
gc.collect()
print('Features made')
train = normalize(featured)
del featured
gc.collect()
print('Normalization made')
print('Training data ready')
gc.collect()

print('Making model')
model = Sequential()
model.add(Dense(train.shape[1], input_shape=(train.shape[1],), kernel_initializer='he_uniform', bias_initializer='he_uniform', activation='sigmoid'))
model.add(Dense(32, activation='relu', kernel_initializer='he_uniform', bias_initializer='he_uniform'))
model.add(Dropout(0.3, seed=seed))
model.add(Dense(8, activation='relu', kernel_initializer='he_uniform', bias_initializer='he_uniform'))
model.add(Dropout(0.3, seed=seed))
model.add(Dense(4, activation='relu', kernel_initializer='he_uniform', bias_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #mean_squared_error
gc.collect()
print('Model ready')

print('Training model')
class_weight = {0:.01,1:.99}
model.fit(train, y, epochs=30, batch_size=3000, verbose=2, validation_split=0.3, class_weight=class_weight, shuffle=True)
gc.collect()
print('Trained model')

print('Loading testing data') 
test_raw = pd.read_csv('../input/test.csv', usecols = test_cols, dtype=col_types)
print('Testing data loaded')
gc.collect()

print('Procesing testing data')
output = pd.DataFrame()
output['click_id'] = test_raw['click_id']
test_raw.drop(['click_id'], axis=1, inplace=True)
test_featured = features(test_raw)
del test_raw
gc.collect()
print('Features made')
test = normalize(test_featured)
del test_featured
gc.collect()
print('Normalization made')
print('Testing data processed')
print('Test data ready')

print('Making prediction')
output['is_attributed'] = model.predict(test, verbose=2)
gc.collect()

print('Saving data')
output.to_csv('answer-' + str(time.time()) + '.csv', float_format='%.8f', index=False)
print('Saved data')
# Any results you write to the current directory are saved as output.
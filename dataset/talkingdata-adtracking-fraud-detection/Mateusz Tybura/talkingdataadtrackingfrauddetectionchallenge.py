import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import gc
import os

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

n_rows_to_skip = 110000000
print('Loading training data')
train_raw = pd.read_csv('../input/train.csv', usecols = train_cols, dtype=col_types, skiprows = range(1, n_rows_to_skip))
print('Training data loaded')
gc.collect()

print('Processing training data')
train = features(train_raw)
y = train['is_attributed']
train.drop(['is_attributed'], axis=1, inplace=True)
print('Training data ready')
gc.collect()

print('Making train matrix')
dtrain = lgb.Dataset(train.values, label=y.values)
del train, y
gc.collect()
print('Train matrix ready')

print('Making model') 
params = {
    'boosting_type': 'gbdt',  # I think dart would be better, but takes too long to run
    # 'drop_rate': 0.09,  # Rate at which to drop trees
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 11,  # Was 255: Reduced to control overfitting
    'max_depth': -1,  # Was 8: LightGBM splits leaf-wise, so control depth via num_leaves
    'min_child_samples': 100,
    'max_bin': 100,
    'subsample': 0.9,  # Was 0.7
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
    'subsample_for_bin': 200000,
    'min_split_gain': 0,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'nthread': 4,
    'verbose': 0,
    'scale_pos_weight': 99.76  # Closer to ratio of positives in train set
}
model = lgb.train(params=params, train_set=dtrain, num_boost_round=300)
del dtrain
gc.collect()
print('Trained model')
print('Feature importances:', list(model.feature_importance()))

print('Loading testing data') 
test_raw = pd.read_csv('../input/test.csv', usecols = ['click_id'], dtype=col_types)
print('Testing data loaded')
gc.collect()

print('Procesing testing data')
test = features(test_raw)
del test_raw
gc.collect()
print('Testing data processed')

output = pd.DataFrame()
output['click_id'] = test['click_id']
test.drop(['click_id'], axis=1, inplace=True)
gc.collect()
print('Test data ready')

print('Making test matrix') 
dtest = lgb.Dataset(test.values)
del test
gc.collect()
print('Test matrix ready')

print('Making prediction')

output['is_attributed'] = model.predict(dtest, ntree_limit=model.best_ntree_limit)
del dtest
gc.collect()

print('Saving data')
output.to_csv('answer.csv', float_format='%.8f', index=False)
print('Saved data')
# Any results you write to the current directory are saved as output.
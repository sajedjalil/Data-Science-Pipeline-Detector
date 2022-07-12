# Any results you write to the current directory are saved as output.
import pandas as pd
import time

# Data Loading from csv file
path = '../input/'
start_time = time.time()
columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }
        
#train = pd.read_csv(path+"train_sample.csv", usecols=columns, dtype=dtypes)
train=pd.read_csv(path+"train.csv", skiprows=range(1,149903891), nrows=30000000, usecols=columns, dtype=dtypes)
test = pd.read_csv(path+"test.csv")
print('Data loading is completed : [{}] seconds'.format(time.time() - start_time))


# split data to input and target
y = train['is_attributed']
train.drop(['is_attributed'], axis=1, inplace=True)
sub = pd.DataFrame()
sub['click_id'] = test['click_id']
test.drop('click_id', axis=1, inplace=True)

# Some feature engineering
nrow_train = train.shape[0]
merge = pd.concat([train, test])


# click_time split to day of weed, hour, minutes, seconds
def datatimeFeatures(df):
    df['datetime'] = pd.to_datetime(df['click_time'])
    df['dow'] = df['datetime'].dt.dayofweek
    df['click_quarter'] = df['datetime'].dt.quarter
    df['click_hour'] = df['datetime'].dt.hour
    df['click_minute'] = df['datetime'].dt.minute
    df['click_second']  = df['datetime'].dt.second
    df.drop(['click_time', 'datetime'], axis=1, inplace=True)
    return df

merge = datatimeFeatures(merge)

# Count the number of clicks by ip in app
ip_count_app = merge.groupby('ip')['app'].count().reset_index()
ip_count_app.columns = ['ip', 'clicks_by_ip_app']
merge = pd.merge(merge, ip_count_app, on='ip', how='left', sort=False)

# Count the number of clicks by ip in os
#ip_count_os = merge.groupby('ip')['os'].count().reset_index()
#ip_count_os.columns = ['ip', 'clicks_by_ip_os']
#merge = pd.merge(merge, ip_count_os, on='ip', how='left', sort=False)

# Count the number of clicks by device
#ip_count_device = merge.groupby('ip')['device'].count().reset_index()
#ip_count_device.columns = ['ip', 'clicks_by_ip_device']
#merge = pd.merge(merge, ip_count_device, on='ip', how='left', sort=False)

# Count the number of clicks by channel
#ip_count_channel = merge.groupby('ip')['channel'].count().reset_index()
#ip_count_channel.columns = ['ip', 'clicks_by_ip_channel']
#merge = pd.merge(merge, ip_count_channel, on='ip', how='left', sort=False)

# drop ip column
merge.drop('ip', axis=1, inplace=True)
print('preprocessing is completed in [{}] seconds'.format(time.time() - start_time))

# split to train and test data
train = merge[:nrow_train]
test = merge[nrow_train:]

# Training model
from sklearn.model_selection import train_test_split
import xgboost as xgb

params = {'eta': 0.9,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,
          'max_depth': 0,
          'subsample': 0.9,
          'colsample_bytree': 0.7,
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'binary:logistic',
          'scale_pos_weight':9,
          'eval_metric': 'auc',
          'nthread':8,
          'random_state': 84,
          'silent': True}

watchlist = [(xgb.DMatrix(train, y), 'train')]
model = xgb.train(params, xgb.DMatrix(train, y), 15, watchlist, maximize=True, verbose_eval=1)
print('XGBoost Training is finished in [{}] seconds'.format(time.time() - start_time))

# prediction and Result saving
sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
sub.to_csv('xgb_submission_row.csv',index=False)
print('submission is done in [{}] seconds'.format(time.time() - start_time))


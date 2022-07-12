import datetime
import lightgbm as lgb
import numpy as np
import os
import pandas as pd
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

# Load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
ss = pd.read_csv('../input/sample_submission.csv')

def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log1p(y_true + 1) - np.log1p(y_pred + 1), 2)))

def extract_features(df):
    df['distance'] = np.sqrt(np.power(df['dropoff_longitude'] - df['pickup_longitude'], 2) + np.power(df['dropoff_latitude'] - df['pickup_latitude'], 2))
    df['month'] = df['pickup_datetime'].apply(lambda x: int(x.split(' ')[0].split('-')[1]))
    df['day'] = df['pickup_datetime'].apply(lambda x: int(x.split(' ')[0].split('-')[2]))
    df['hour'] = df['pickup_datetime'].apply(lambda x: int(x.split(' ')[1].split(':')[0]))
    df['minutes'] = df['pickup_datetime'].apply(lambda x: int(x.split(' ')[1].split(':')[1]))
    df['is_weekend'] = ((df.pickup_datetime.astype('datetime64[ns]').dt.dayofweek) // 4 == 1).astype(float)
    df['weekday'] = df.pickup_datetime.astype('datetime64[ns]').dt.dayofweek
    df['is_holyday'] = df.apply(lambda row: 1 if (row['month']==1 and row['day']==1) or (row['month']==7 and row['day']==4) or (row['month']==11 and row['day']==11) or (row['month']==12 and row['day']==25) or (row['month']==1 and row['day'] >= 15 and row['day'] <= 21 and row['weekday'] == 0) or (row['month']==2 and row['day'] >= 15 and row['day'] <= 21 and row['weekday'] == 0) or (row['month']==5 and row['day'] >= 25 and row['day'] <= 31 and row['weekday'] == 0) or (row['month']==9 and row['day'] >= 1 and row['day'] <= 7 and row['weekday'] == 0) or (row['month']==10 and row['day'] >= 8 and row['day'] <= 14 and row['weekday'] == 0) or (row['month']==11 and row['day'] >= 22 and row['day'] <= 28 and row['weekday'] == 3) else 0, axis=1)
    df['is_day_before_holyday'] = df.apply(lambda row: 1 if (row['month']==12 and row['day']==31) or (row['month']==7 and row['day']==3) or (row['month']==11 and row['day']==10) or (row['month']==12 and row['day']==24) or (row['month']==1 and row['day'] >= 14 and row['day'] <= 20 and row['weekday'] == 6) or (row['month']==2 and row['day'] >= 14 and row['day'] <= 20 and row['weekday'] == 6) or (row['month']==5 and row['day'] >= 24 and row['day'] <= 30 and row['weekday'] == 6) or ((row['month']==9 and row['day'] >= 1 and row['day'] <= 6) or (row['month']==8 and row['day'] == 31) and row['weekday'] == 6) or (row['month']==10 and row['day'] >= 7 and row['day'] <= 13 and row['weekday'] == 6) or (row['month']==11 and row['day'] >= 21 and row['day'] <= 27 and row['weekday'] == 2) else 0, axis=1)
    df.drop('day', axis=1, inplace=True)

# Extract features
print('Extracting train features')
extract_features(train)
print('Extracting test features')
extract_features(test)

print(train.head())

# Prepare data
X = np.array(train.drop(['id', 'pickup_datetime', 'dropoff_datetime', 'store_and_fwd_flag', 'trip_duration'], axis=1))
y = np.log1p(train['trip_duration'].values)
mean_trip_duration = np.mean(np.exp(y) - 1 )

print('X.shape = ' + str(X.shape))
print('y.shape = ' + str(y.shape))

X_test = np.array(test.drop(['id', 'pickup_datetime', 'store_and_fwd_flag'], axis=1))

print('X_test.shape = ' + str(X_test.shape))

print('Training and making predictions')
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmsle',
    'max_depth': 6, 
    'learning_rate': 0.1,
    'verbose': 0, 
    'early_stopping_round': 20}
n_estimators = 100

n_iters = 5
preds_buf = []
err_buf = []
for i in range(n_iters): 
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=i)
    d_train = lgb.Dataset(x_train, label=y_train)
    d_valid = lgb.Dataset(x_valid, label=y_valid)
    watchlist = [d_valid]

    model = lgb.train(params, d_train, n_estimators, watchlist, verbose_eval=1)

    preds = np.exp(model.predict(x_valid)) - 1
    preds[preds < 0] = mean_trip_duration
    err = rmsle(np.exp(y_valid) - 1, preds)
    err_buf.append(err)
    print('RMSLE = ' + str(err))
    
    preds = np.exp(model.predict(X_test)) - 1
    preds[preds < 0] = mean_trip_duration
    preds_buf.append(preds)

print('Mean RMSLE = ' + str(np.mean(err_buf)) + ' +/- ' + str(np.std(err_buf)))
# Average predictions
preds = np.mean(preds_buf, axis=0)

# Prepare submission
subm = pd.DataFrame()
subm['id'] = test.id.values
subm['trip_duration'] = preds
subm.to_csv('submission_lgbm.csv', index=False)

# coding: utf-8
# This code is inspired by the following kernels:
# https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-lb-0-9680
# https://www.kaggle.com/aharless/talkingdata-time-deltas
import pandas as pd
import numpy as np
import gc
import psutil
import os
import joblib

from catboost import CatBoostClassifier

# As in:
# https://www.kaggle.com/aharless/talkingdata-time-deltas
process = psutil.Process(os.getpid())


def print_ram_use():
    print('Total memory in use: {0:.2f} GB\n'.format(process.memory_info().rss/(2**30)))


# Definitions
TARGET_COL = 'is_attributed'
TRAIN_COLS = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
TEST_COLS = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']

# Load the train set
# We keep "only" the last 40M rows, as in:
# https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-lb-0-9680
dtypes_train = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
}

print('Loading train dataset ...')
df_train = pd.read_csv('../input/train.csv', dtype=dtypes_train, parse_dates=['click_time'],
                       usecols=TRAIN_COLS + [TARGET_COL],skiprows=range(1, 182903891), nrows=2000000)
print_ram_use()

# Load the test
dtypes_test = {
    'click_id': 'uint32',
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
}
print('Loading test dataset ...')
df_test = pd.read_csv('../input/test.csv', dtype=dtypes_test, parse_dates=['click_time'], usecols=TEST_COLS)
print_ram_use()

# Append the 2 datasets
i_train = len(df_train)
n_val = 2000000
print('Concatenate train and test datasets ...')
df_tot = df_train.append(df_test)
del df_test
gc.collect()
print_ram_use()

# Transform click_time to unix time
# from: https://www.kaggle.com/aharless/talkingdata-time-deltas
df_tot['click_time'] = df_tot['click_time'].astype('int64').floordiv(1.E+09).astype('int32')
print_ram_use()


# Functions for feature extraction
# most of this is inspired by:
# https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-lb-0-9680
def add_time_features(df):
    df['hour'] = pd.to_datetime(df['click_time']).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df['click_time']).dt.day.astype('uint8')
    df['month'] = pd.to_datetime(df['click_time']).dt.month.astype('uint8')
    df['minute'] = pd.to_datetime(df['click_time']).dt.minute.astype('uint8')


def group_agg_merge(df, by, agg, what, name):
    if agg == 'count':
        tmp = df[['ip', 'app', 'device', 'os', 'channel', 'minute', 'hour', 'day', 'month']].groupby(by=by)[[what]].count().reset_index().rename(index=str, columns={what: name})
    elif agg == 'mean':
        tmp = df[['ip', 'app', 'device', 'os', 'channel', 'minute', 'hour', 'day', 'month']].groupby(by=by)[[what]].mean().reset_index().rename(index=str, columns={what: name})
    elif agg == 'var':
        tmp = df[['ip', 'app', 'device', 'os', 'channel', 'minute', 'hour', 'day', 'month']].groupby(by=by)[[what]].var().reset_index().rename(index=str, columns={what: name})
    else:
        raise NotImplementedError('Aggregation type %s is not supported' % agg)

    df = df.merge(tmp, on=by, how='left')
    del tmp
    gc.collect()
    
    return df


# Add time features
print('Adding time features ...')
add_time_features(df_tot)
gc.collect()
print_ram_use()

# Define the aggregation we want to perform
agg_dict = {
    'ip_day_hour_count': (['ip', 'day', 'hour'], 'count', 'channel'),
    'ip_day_hour_minute_count': (['ip', 'day', 'hour', 'minute'], 'count', 'channel'),
    'ip_day_count': (['ip', 'day'], 'count', 'channel'),
    'app_os_count': (['app', 'os'], 'count', 'channel'),
    'app_channel_count': (['app', 'channel'], 'count', 'os'),
    'ip_app_count': (['ip', 'app'], 'count', 'channel'),
    'ip_app_os_count': (['ip', 'app', 'os'], 'count', 'channel'),
    'ip_day_channel_var_hour': (['ip', 'day', 'channel'], 'var', 'hour'),
    'ip_app_os_var_hour': (['ip', 'app', 'os'], 'var', 'hour'),
    'ip_app_channel_var_day': (['ip', 'app', 'channel'], 'var', 'day'),
    'ip_app_channel_mean_hour': (['ip', 'app', 'channel'], 'mean', 'hour'),
    'ip_app_channel_mean_month': (['ip', 'app', 'channel'], 'mean', 'month'),
}

print('Adding aggregation features ...')
for name, (by, agg, what) in agg_dict.items():
    print('%s of %s by %s' % (agg, what, by))
    df_tot = group_agg_merge(df_tot, by, agg, what, name)
    gc.collect()
    print_ram_use()

# Let's look at the new features we have added
df_tot.head(10)

# Let's try to reduce the memory in use
print('Saving some memory ...')
print_ram_use()
df_tot = df_tot.fillna(-1)
df_tot['ip_day_hour_count'] = df_tot['ip_day_hour_count'].astype('int16')
df_tot['ip_app_count'] = df_tot['ip_app_count'].astype('int16')
df_tot['ip_app_os_count'] = df_tot['ip_app_os_count'].astype('int16')
print_ram_use()

# Divide the dataset back to train/val/test
print_ram_use()
print('Splitting train/val/test ...')
df_train = df_tot[:i_train]
df_val = df_train[-n_val:]
df_train = df_train[:-n_val]
df_test = df_train[i_train:]
del df_tot
gc.collect()
print_ram_use()
print(len(df_train), len(df_val))

# List of features
PREDICTORS = ['app', 'os', 'device', 'channel', 
              'ip_day_hour_count', 'ip_day_count', 'ip_app_count', 'app_channel_count', 'ip_app_os_count', 'ip_day_hour_minute_count',
              'ip_day_channel_var_hour', 'ip_app_os_var_hour', 'ip_app_channel_var_day']
CATEGORICAL = ['app', 'device', 'os', 'channel']


# Catboost
print('Fitting Catboost...')
clf = CatBoostClassifier(learning_rate=0.15,
                         loss_function='Logloss',
                         thread_count=4,
                         use_best_model=True,
                         verbose=True,
                         eval_metric='AUC',
                         max_depth=5,
                         n_estimators=40,
                         random_state=42,
                         max_bin=100,
                         scale_pos_weight=100
                         )
clf.fit(X=df_train[PREDICTORS].as_matrix(), y=df_train[TARGET_COL].values, cat_features=list(range(4)),
        eval_set=(df_val[PREDICTORS].as_matrix(), df_val[TARGET_COL].values), verbose=True)

print('Feature importances:')
fimport = clf.get_feature_importance(X=df_train[PREDICTORS].as_matrix(), y=df_train[TARGET_COL].values,
                                     cat_features=list(range(4)))
for fname, fimp in zip(PREDICTORS, fimport):
    print(' {0} = {1:.3f}'.format(fname, fimp))

# Write submission
print('Writing submission file ...')
sub = pd.DataFrame()
sub['click_id'] = df_test['click_id'].astype('int')
sub['is_attributed'] = np.round(clf.predict_proba(df_test[PREDICTORS])[:, 1], 4)
sub.to_csv('catb.csv', index=False)

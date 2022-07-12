"""
Acknowledgements:
    - This kernel was forked from Alexey Pronin's LightGBM code, so many thanks
      for his work:
          https://www.kaggle.com/graf10a/lightgbm-lb-0-9675/code

What's new:
    - Creation of a feature processing function to remove the need to append the
      test and train sets for processing. I therefore don't need the (huge) test
      set in memory whilst training the LightGBM model, and as such can load
      more training data --> better generalisation.

    - Use of (optional) cross validation instead of a validation set to know when to 
      stop boosting. This allows me to train on the full training set and also checks
      performance against more than one fold --> even better generalisation.

    - Together, these two changes let me train on ~75m rows.

    - Additional features tested:
        - (REJECTED) n_ip_clicks: Count of clicks per IP address
        - (REJECTED) day_section: 'hour' binned into morning, work hours etc.

    - Hyperparameter optimisation. Changes are described where the lightgbm
      parameters are set below.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
os.environ['OMP_NUM_THREADS'] = '4'  # Number of threads on the Kaggle server

"""
0. Run Params
"""
run_cv = False  # Run CV to get opt boost rounds
OPT_BOOST_ROUNDS = 349  # Found through CV on my machine to save Kaggle server time

"""
1. Load data
"""

path_train = os.path.join(os.pardir, 'input', 'train.csv')
path_test = os.path.join(os.pardir, 'input', 'test.csv')

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']

dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}

n_rows_in_train_set = 184903891  # Roughly
n_rows_to_skip = 110000000

print('Loading the training data...')
train = pd.read_csv(path_train, dtype=dtypes, skiprows = range(1, n_rows_to_skip), usecols=train_cols)

len_train = len(train)
print('The initial size of the train set is', len_train)
gc.collect()

"""
2. Feature engineering function
"""

def process_data(df):

    print("Creating new time features: 'hour', 'day'")
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    day_section = 0
    for start_time, end_time in zip([0, 6, 12, 18], [6, 12, 18, 24]):
        df.loc[(df['hour'] >= start_time) & (df['hour'] < end_time), 'day_section'] = day_section
        day_section += 1
    df['day_section'] = df['day_section'].astype('uint8')
    gc.collect()

    print("Creating new click count features...")

    print('Computing the number of clicks associated with a given IP address...')
    ip_clicks = df[['ip','channel']].groupby(by=['ip'])[['channel']]\
        .count().reset_index().rename(columns={'channel': 'n_ip_clicks'})

    print('Merging the IP clicks data with the main data set...')
    df = df.merge(ip_clicks, on=['ip'], how='left')
    del ip_clicks
    gc.collect()
    
    print('Computing the number of clicks associated with a given app per hour...')
    app_clicks = df[['app', 'day', 'hour', 'channel']].groupby(by=['app', 'day', 'hour'])[['channel']]\
        .count().reset_index().rename(columns={'channel': 'n_app_clicks'})

    print('Merging the IP clicks data with the main data set...')
    df = df.merge(app_clicks, on=['app', 'day', 'hour'], how='left')
    del app_clicks
    gc.collect()


    print('Computing the number of channels associated with\n'
          'a given IP address within each hour...')
    n_chans = df[['ip','day','hour','channel']].groupby(by=['ip','day',
              'hour'])[['channel']].count().reset_index().rename(columns={'channel': 'n_channels'})

    print('Merging the channels data with the main data set...')
    df = df.merge(n_chans, on=['ip','day','hour'], how='left')
    del n_chans
    gc.collect()

    print('Computing the number of channels associated with ')
    print('a given IP address and app...')
    n_chans = df[['ip','app', 'channel']].groupby(by=['ip',
              'app'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_count'})

    print('Merging the channels data with the main data set...')
    df = df.merge(n_chans, on=['ip','app'], how='left')
    del n_chans
    gc.collect()

    print('Computing the number of channels associated with ')
    print('a given IP address, app, and os...')
    n_chans = df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app',
              'os'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_os_count'})

    print('Merging the channels data with the main data set...')
    df = df.merge(n_chans, on=['ip','app', 'os'], how='left')
    del n_chans
    gc.collect()

    print("Adjusting the data types of the new count features... ")
    df.info()
    for feat in ['n_channels', 'ip_app_count', 'ip_app_os_count', 'n_ip_clicks']:
        df[feat] = df[feat].astype('uint16')

    return df

"""
3. Training & validation
"""

# Apply processing function to the train set
train = process_data(df=train)

target = 'is_attributed'
train[target] = train[target].astype('uint8')
train.info()

predictors = ['ip', 'device', 'app', 'os', 'channel', 'hour', 'n_channels',
    'ip_app_count', 'ip_app_os_count']
categorical = ['ip', 'app', 'device', 'os', 'channel', 'hour']
gc.collect()

print("Preparing the datasets for training...")
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
    'scale_pos_weight': 99.76
}

dtrain = lgb.Dataset(train[predictors].values, label=train[target].values,
                      feature_name=predictors,
                      categorical_feature=categorical
                      )
del train
gc.collect()
print('Datasets ready.')

if run_cv:
    print('Cross validating...')
    cv_results = lgb.cv(params=params,
                        train_set=dtrain,
                        nfold=3,
                        num_boost_round=350,
                        early_stopping_rounds=30,
                        verbose_eval=20,
                        categorical_feature=categorical)
    OPT_BOOST_ROUNDS = np.argmax(cv_results['auc-mean'])
    print('CV complete. Optimum boost rounds = {}'.format(OPT_BOOST_ROUNDS))

print('Training model...')
lgb_model = lgb.train(params=params, train_set=dtrain, num_boost_round=OPT_BOOST_ROUNDS,
                      categorical_feature=categorical)
print('Model trained.')

# Feature names:
print('Feature names:', lgb_model.feature_name())

# Feature importances:
print('Feature importances:', list(lgb_model.feature_importance()))

"""
4. Infer and submit
"""

# Clear up after training
del dtrain
gc.collect()

print('Loading the test data...')
test = pd.read_csv(path_test, dtype=dtypes, header=0, usecols=test_cols)
test = process_data(test)

print("Preparing data for submission...")
submit = pd.read_csv(path_test, dtype='int', usecols=['click_id'])

print("Predicting the submission data...")
submit['is_attributed'] = lgb_model.predict(test[predictors], num_iteration=lgb_model.best_iteration)

print("Writing the submission data into a csv file...")
submit.to_csv('submission.csv', index=False)

print("Writing complete.")

# Alexey Pronin's tranlation of Krishna's LGBM into Python. 
# Big thanks to Andy Harless for his Pranav's 
# LGBM Python code -- I used it as a starter code. 
# Big thank you to Pranav and all others who contributed!
#############################################################################
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split 
import lightgbm as lgb
import gc
##############################################################################
TEST = False # True #
TEST_SIZE = 100000
VALIDATE = True # False #
MAX_ROUNDS = 1500
EARLY_STOP = 50
OPT_ROUNDS = 500
##############################################################################
total_rows = 184903890
##############################################################################
if TEST:
    rows_train = TEST_SIZE
    rows_test = TEST_SIZE
    skip_train = None
else:
    rows_train = 40000000
    rows_test = None 
    skip_train = range(1, total_rows - rows_train + 1)
##############################################################################
path = '../input/'
path_train = path + 'train.csv'
path_test = path + 'test.csv'
##############################################################################
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
##############################################################################
print('Load train...')
train_cols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed'] 
train = pd.read_csv(path_train, skiprows=skip_train, nrows=rows_train, 
                        dtype=dtypes, usecols=train_cols)
##############################################################################
gc.collect()
##############################################################################
print('Building features...')
most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]
##############################################################################
def prep_data( df ):
    ##############################################################################
    print('hour')
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    print('day')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df.drop(['click_time'], axis=1, inplace=True)
    gc.collect()
    ##############################################################################
    print('in_test_hh')
    df['in_test_hh'] = (   2 
                         - 1*df['hour'].isin(  most_freq_hours_in_test_data ) 
                         + 1*df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')
    print( df.info() )
    ##############################################################################
    print('group by : n_ip_day_test_hh')                                                        
    gp = df[['ip', 'day', 'in_test_hh', 'channel']].groupby(by=['ip', 'day',
            'in_test_hh'])[['channel']].count().reset_index().rename(index=str, 
            columns={'channel': 'n_ip_day_test_hh'})
    df = df.merge(gp, on=['ip','day','in_test_hh'], how='left')
    del gp
    df.drop(['in_test_hh'], axis=1, inplace=True)
    df['n_ip_day_test_hh'] = df['n_ip_day_test_hh'].astype('uint32')
    gc.collect()
    print( df.info() )
    ##############################################################################
    print('group by : n_ip_day_hh')
    gp = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 
            'hour'])[['channel']].count().reset_index().rename(index=str, 
            columns={'channel': 'n_ip_day_hh'})
    df = df.merge(gp, on=['ip', 'day', 'hour'], how='left')
    del gp
    df['n_ip_day_hh'] = df['n_ip_day_hh'].astype('uint16')
    gc.collect()
    print( df.info() )
    ##############################################################################
    print('group by : n_ip_os_day_hh')
    gp = df[['ip', 'os', 'day', 'hour', 'channel']].groupby(by=['ip', 'os', 'day',
            'hour'])[['channel']].count().reset_index().rename(index=str, 
            columns={'channel': 'n_ip_os_day_hh'})
    df = df.merge(gp, on=['ip','os', 'day', 'hour'], how='left')
    del gp
    df['n_ip_os_day_hh'] = df['n_ip_os_day_hh'].astype('uint16')
    gc.collect()
    print( df.info() )
    ##############################################################################
    print('group by : n_ip_app_day_hh')
    gp = df[['ip', 'app', 'day', 'hour', 'channel']].groupby(by=['ip', 'app', 'day',
            'hour'])[['channel']].count().reset_index().rename(index=str, 
            columns={'channel': 'n_ip_app_day_hh'})
    df = df.merge(gp, on=['ip', 'app', 'day', 'hour'], how='left')
    del gp
    df['n_ip_app_day_hh'] = df['n_ip_app_day_hh'].astype('uint16')
    gc.collect()
    print( df.info() )
    ##############################################################################
    print('group by : n_ip_app_os_day_hh')
    gp = df[['ip', 'app', 'os', 'day', 'hour', 'channel']].groupby(by=['ip', 'app', 'os', 
            'day', 'hour'])[['channel']].count().reset_index().rename(index=str, 
            columns={'channel': 'n_ip_app_os_day_hh'})
    df = df.merge(gp, on=['ip', 'app', 'os', 'day', 'hour'], how='left')
    del gp
    df['n_ip_app_os_day_hh'] = df['n_ip_app_os_day_hh'].astype('uint32')
    gc.collect()
    print( df.info() )
    ##############################################################################
    print('group by : n_app_day_hh')
    gp = df[['app', 'day', 'hour', 'channel']].groupby(by=['app', 
            'day', 'hour'])[['channel']].count().reset_index().rename(index=str, 
            columns={'channel': 'n_app_day_hh'})
    df = df.merge(gp, on=['app', 'day', 'hour'], how='left')
    del gp
    df['n_app_day_hh'] = df['n_app_day_hh'].astype('uint32')
    gc.collect()
    print( df.info() )
    ##############################################################################
    df.drop( ['day'], axis=1, inplace=True )
    gc.collect()
    print( df.info() )
    ##############################################################################
    print('group by : n_ip_app_dev_os')
    gp = df[['ip', 'app', 'device', 'os', 'channel']].groupby(by=['ip', 'app', 
             'device', 'os'])[['channel']].count().reset_index().rename(index=str, 
            columns={'channel': 'n_ip_app_dev_os'})
    df = df.merge(gp, on=['ip', 'app', 'device', 'os'], how='left')
    del gp
    df['n_ip_app_dev_os'] = df['n_ip_app_dev_os'].astype('uint32')
    gc.collect()
    print( df.info() )
    ##############################################################################
    df['ip_app_dev_os_cumcount'] = df.groupby(['ip', 'app', \
                                                'device', 'os']).cumcount().astype('uint16')
    ##############################################################################
    print('group by : n_ip_dev_os')
    gp = df[['ip', 'device', 'os', 'channel']].groupby(by=['ip', 'device', 
             'os'])[['channel']].count().reset_index().rename(index=str, 
            columns={'channel': 'n_ip_dev_os'})
    df = df.merge(gp, on=['ip', 'device', 'os'], how='left')
    del gp
    df['n_ip_dev_os'] = df['n_ip_dev_os'].astype('uint32')
    gc.collect()
    print( df.info() )
    ##############################################################################
    df['ip_dev_os_cumcount'] = df.groupby(['ip', 'device', 'os']).cumcount().astype('uint16')
    ##############################################################################
    df.drop( ['ip'], axis=1, inplace=True )
    gc.collect()
    print( df.info() )
    ##############################################################################
    return( df )
##############################################################################
print( "Train info before processing: ")
train.info()
##############################################################################
train = prep_data( train )
gc.collect()
##############################################################################
print("Variables and data type: ")
train.info()
##############################################################################
metrics = 'auc'
##############################################################################
lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':metrics,
        'learning_rate': 0.1,
        'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 4,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'nthread': 4,
        'verbose': 0,
        'scale_pos_weight':99.7, # because training data is extremely unbalanced 
}
##############################################################################
target = 'is_attributed'
##############################################################################
predictors = list(set(train.columns) - {'is_attributed', 'click_time'})
print('The list of predictors:')
for item in predictors: print(item)
##############################################################################
categorical = ['app', 'device', 'os', 'channel', 'hour']
##############################################################################
print(train.head(5))
##############################################################################
if VALIDATE:
    ##############################################################################
    train, val = train_test_split( train, train_size=.95, random_state=99, shuffle=False )
    ##############################################################################
    print(train.info())
    print(val.info())
    ##############################################################################
    print("train size: ", len(train))
    print("valid size: ", len(val))
    ##############################################################################
    gc.collect()
    ##############################################################################
    print("Training...")
    ##############################################################################
    num_boost_round=MAX_ROUNDS
    early_stopping_rounds=EARLY_STOP
    ##############################################################################
    dtrain = lgb.Dataset(train[predictors].values, label=train[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del train
    gc.collect()
    ##############################################################################
    dvalid = lgb.Dataset(val[predictors].values, label=val[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del val
    gc.collect()
    ##############################################################################
    evals_results = {}
    ##############################################################################
    bst = lgb.train(lgb_params, 
                     dtrain, 
                     valid_sets=[dvalid], 
                     valid_names=['valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=50, 
                     feval=None)
    ##############################################################################
    n_estimators = bst.best_iteration
    ##############################################################################
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])
    ##############################################################################
    del dvalid
    ##############################################################################
else:
    ##############################################################################
    print(train.info())
    ##############################################################################
    print("train size: ", len(train))
    ##############################################################################
    gc.collect()
    ##############################################################################
    print("Training...")
    ##############################################################################
    num_boost_round=OPT_ROUNDS
    ##############################################################################
    dtrain = lgb.Dataset(train[predictors].values, label=train[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del train
    gc.collect()
    ##############################################################################
    evals_results = {}
    ##############################################################################
    bst = lgb.train(lgb_params, 
                     dtrain, 
                     valid_sets=[dtrain], 
                     valid_names=['train'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     verbose_eval=50, 
                     feval=None)
    ##############################################################################
    n_estimators = num_boost_round
    ##############################################################################
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['train'][metrics][n_estimators-1])
    ##############################################################################
del dtrain
gc.collect()
##############################################################################
print('Loading test...')
test_cols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']
test = pd.read_csv(path_test, nrows=rows_test, dtype=dtypes, usecols=test_cols)
##############################################################################
test = prep_data( test )
gc.collect()
##############################################################################
sub = pd.DataFrame()
sub['click_id'] = test['click_id']
##############################################################################
print("Predicting...")
sub['is_attributed'] = bst.predict(test[predictors])
##############################################################################
print("Writing prediction to a csv file...")
sub.to_csv('LGBM_python.csv', index=False)
print(sub.info())
##############################################################################
print("All done!..")
##############################################################################
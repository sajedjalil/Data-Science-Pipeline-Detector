#!/usr/bin/env python

# This is a port to CatBoost + undersampling of 
# Alexey Pronin's tranlation of Krishna's LGBM into Python. 
# Big thanks to Andy Harless for his Pranav's 
# LGBM Python code -- I used it as a starter code. 
# Big thank you to Pranav and all others who contributed!
#############################################################################
from distutils.dir_util import mkpath

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split 
import catboost as cb
import gc
import operator
from imblearn.under_sampling import RandomUnderSampler 
##############################################################################
out_suffix = '_CB_1_1'

TEST = False
TEST_SIZE = 100000
VALIDATE = True
MAX_ROUNDS = 1500
EARLY_STOP = 100
OPT_ROUNDS = 1200

USED_RAM_LIMIT = 5*(2 ** 30)

CLASSES_RATIO = 1.0
##############################################################################
total_rows = 184903890
##############################################################################
if TEST:
    rows_train = TEST_SIZE
    rows_test = TEST_SIZE
    skip_train = None
    comp_suffix = '.test'
else:
    rows_train = 40000000
    rows_test = None 
    skip_train = range(1, total_rows - rows_train + 1)
    comp_suffix = ''
##############################################################################
input_path = '../input/'

path_train = input_path + 'train.csv'
path_test = input_path + 'test.csv'


intermed_path = './intermed' + comp_suffix + '/'

output_path = './'

mkpath(intermed_path)
mkpath(output_path)


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
eval_metric = 'AUC'
##############################################################################
cb_params = {
    # technical
    'verbose' : True,
    'random_seed' : 42,
    'save_snapshot' : True,
    'snapshot_file' : output_path + 'snapshot',
    'used_ram_limit' : USED_RAM_LIMIT,
    
    # learning
    'l2_leaf_reg' : 150,
    'scale_pos_weight' : CLASSES_RATIO,
    'one_hot_max_size' : 100,
    'max_ctr_complexity' : 3,
    'leaf_estimation_iterations' : 8,
    'learning_rate' : 0.1,
    'eval_metric' : eval_metric
}
##############################################################################
target = 'is_attributed'
##############################################################################
predictors = list(set(train.columns) - {'is_attributed', 'click_time'})
print('The list of predictors:')
for item in predictors: print(item)
##############################################################################
categorical = ['app', 'device', 'os', 'channel', 'hour']

categorical_features_indices = [predictors.index(cat_name) for cat_name in categorical]

print('categorical_features_indices', categorical_features_indices)
##############################################################################
print(train.head(5))
##############################################################################
alltrain_X = train[predictors].values
alltrain_Y = train[target].values


if VALIDATE:
    ##############################################################################
    
    train_X, val_X, train_Y, val_Y = train_test_split(
        alltrain_X, alltrain_Y, train_size=.95, random_state=99, shuffle=True,
        stratify = alltrain_Y
    )
    ##############################################################################
    print("train shape: ", train_X.shape)
    print("valid shape: ", val_X.shape)
    ##############################################################################
    gc.collect()

    ##############################################################################
    print("Undersampling...")
    ##############################################################################
    target_counts = np.bincount(train_Y)
    print('target_counts', target_counts)
    
    rus = RandomUnderSampler(random_state=42, 
                             ratio={0: int(CLASSES_RATIO*target_counts[1]),
                                    1: target_counts[1]})
    
    uns_train_X, uns_train_Y = rus.fit_sample(train_X, train_Y)
    
    target_counts = np.bincount(train_Y)
    print('target_counts after undersamping', target_counts)
    ##############################################################################
    print("Training...")
    ##############################################################################
    cb_params["iterations"] = MAX_ROUNDS
    cb_params["od_type"] = 'Iter'
    cb_params["od_wait"] = EARLY_STOP
        
    dtrain = cb.Pool(uns_train_X,
                     label=uns_train_Y,
                     feature_names=predictors,
                     cat_features=categorical_features_indices
                    )
    
    dvalid = cb.Pool(val_X,
                     label=val_Y,
                     feature_names=predictors,
                     cat_features=categorical_features_indices
                    )    
    del uns_train_X
    del uns_train_Y
    del val_X
    del val_Y
    gc.collect()
    ##############################################################################
    cb_model = cb.CatBoostClassifier(**cb_params)
    cb_model.fit(dtrain, eval_set=dvalid)
    cb_model.save_model(output_path + "Krishna_s_train_uns_w_valid" + out_suffix + ".cbm")
    
    del dvalid
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
    cb_params['iterations']=OPT_ROUNDS
    ##############################################################################
    dtrain = cb.Pool(alltrain_X, label=alltrain_Y,
                     feature_names=predictors,
                     cat_features=categorical_features_indices
                    )
    del alltrain_X
    del alltrain_Y
    gc.collect()
    ##############################################################################
    cb_model = cb.CatBoostClassifier(**cb_params)
    cb_model.fit(dtrain)
    cb_model.save_model(output_path + "Krishna_s_alltrain_uns_" + out_suffix + ".cbm")
    ##############################################################################

del dtrain
gc.collect()

print('Model params')
print(cb_model.get_params())

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
pred_probs = cb_model.predict_proba(test[predictors].values)
sub['is_attributed'] = [prob[1] for prob in pred_probs]
##############################################################################
print("Writing prediction to a csv file...")
sub.to_csv(output_path + 'Krishna_s_CatBoost_1_1' + out_suffix + '.csv',
           index=False)
print(sub.info())
##############################################################################
print("All done!..")
##############################################################################
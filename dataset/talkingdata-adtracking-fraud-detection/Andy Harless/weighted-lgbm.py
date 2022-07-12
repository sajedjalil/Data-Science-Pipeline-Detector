# Refer to the following kernels:
#    https://www.kaggle.com/sionek/tuning-model-by-pranav-lb-0-9703
#    https://www.kaggle.com/graf10a/pranav-s-lgbm-lb-0-9699
#    https://www.kaggle.com/pranav84/single-lightgbm-in-r-with-75-mln-rows-lb-0-9690
#    https://www.kaggle.com/aharless/try-pranav-s-r-lgbm-in-python
#    https://www.kaggle.com/aharless/weighted-lgb-validation-with-better-early-stopping


VALIDATE = False

MAX_ROUNDS = 3000
EARLY_STOP = 300
OPT_ROUNDS = 420
LEARNING_RATE =.08
NROWS = 75000000
MINWEIGHT = .5
OFF_HRS_DOWNWEIGHT = .5

FULL_OUTFILE = 'sub_lgbm_with_app_nocv.csv'
VALID_OUTFILE = 'sub_lgbm_with_app_withcv.csv'



def do_count( df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )


import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split 
import lightgbm as lgb

path = '../input/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

print('load train...')
train_cols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
skiprows = 184903891 - NROWS
train_df = pd.read_csv(path+"train.csv", skiprows=range(1,skiprows), nrows=NROWS, dtype=dtypes, usecols=train_cols)

import gc

gc.collect()

print('data prep...')

most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]


def prep_data( df ):
    
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df.drop(['click_time'], axis=1, inplace=True)
    gc.collect()
    
    df['in_test_hh'] = (   3 
                         - 2*df['hour'].isin(  most_freq_hours_in_test_data ) 
                         - 1*df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')

    print('group by : ip_day_test_hh')
    
    df = do_count( df, ['ip', 'day', 'in_test_hh'], 'nip_day_test_hh', show_max=True ); gc.collect()
    df = do_count( df, ['ip', 'day', 'hour'], 'nip_day_hh', 'uint16', show_max=True ); gc.collect()
    df = do_count( df, ['ip', 'day', 'os', 'hour'], 'nip_hh_os', 'uint16', show_max=True ); gc.collect()
    df = do_count( df, ['ip', 'day', 'app', 'hour'], 'nip_hh_app', 'uint16', show_max=True ); gc.collect()
    df = do_count( df, ['ip', 'day', 'app', 'os', 'hour'], 'nip_app_os', 'uint16', show_max=True ); gc.collect()
    df = do_count( df, ['app', 'day', 'hour'], 'n_app', 'uint32', show_max=True ); gc.collect()

    df.drop( ['ip','day'], axis=1, inplace=True )
    gc.collect()

    return( df )

#---------------------------------------------------------------------------------

train_df = prep_data( train_df )
gc.collect()

print("vars and data type: ")
train_df.info()

metrics = 'auc'
lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':metrics,
        'learning_rate': LEARNING_RATE,
        'num_leaves': 15, # 7,
        'max_depth': 5, # 4,
        'min_child_samples': 200, # 100, 
        'max_bin': 255, # 100,
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'nthread': 8,
        'verbose': 0,
        'scale_pos_weight':100, # 200,
        'metric':metrics
}

target = 'is_attributed'
predictors = ['app','device','os', 'channel', 'hour', 'nip_day_test_hh', 'nip_day_hh',
              'nip_hh_os', 'nip_hh_app', 'nip_app_os', 'n_app']
categorical = ['app', 'device', 'os', 'channel', 'hour']

print(train_df.head(5))

if VALIDATE:

    train_df, val_df = train_test_split( train_df, train_size=.95, shuffle=False )

    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))

    gc.collect()
    
    weights = MINWEIGHT + (1.-MINWEIGHT)*(train_df.index.values/len(train_df))
    weights *= (1 - OFF_HRS_DOWNWEIGHT*(train_df.in_test_hh.values>1))

    print("Training...")

    num_boost_round=MAX_ROUNDS
    early_stopping_rounds=EARLY_STOP

    xgtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical,
                          weight=weights
                          )
    del train_df
    gc.collect()

    xgvalid = lgb.Dataset(val_df[predictors].values, label=val_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del val_df
    gc.collect()

    evals_results = {}

    bst = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets= [xgvalid], 
                     valid_names=['valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=None)

    n_estimators = bst.best_iteration

    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])
    
    outfile = VALID_OUTFILE
    
    del xgvalid

else:

    print("train size: ", len(train_df))

    gc.collect()
    
    weights = MINWEIGHT + (1.-MINWEIGHT)*(train_df.index.values/len(train_df))
    weights *= (1 - OFF_HRS_DOWNWEIGHT*(train_df.in_test_hh.values>1))

    print("Training...")

    num_boost_round=OPT_ROUNDS

    xgtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical,
                          weight=weights
                          )
    del train_df
    gc.collect()

    bst = lgb.train(lgb_params, 
                     xgtrain, 
                     num_boost_round=num_boost_round,
                     verbose_eval=10, 
                     feval=None)
                     
    outfile = FULL_OUTFILE

del xgtrain
gc.collect()

print('load test...')
test_cols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=test_cols)

test_df = prep_data( test_df )
gc.collect()

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id']

print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])
print("writing...")
sub.to_csv(outfile, index=False, float_format='%.9f')
print("done...")
# With additional memory optimization by Andy Harless,
# based on Nooh's "LGBM_NEW_FEATURES_CORRECTED"
#     https://www.kaggle.com/nuhsikander/lgbm-new-features-corrected
# which was based on Ravi Teja Gutta's "LightGBM with count features"
#     https://www.kaggle.com/rteja1113/lightgbm-with-count-features
# which was based on Pranav Pandya's "LightGBM (Fixing unbalanced data)"
#     https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-auc-0-9787
# which was based on Aloisio Dourado's "lgbm starter"
#     https://www.kaggle.com/aloisiodn/lgbm-starter-early-stopping-0-9539?scriptVersionId=2737977

VALIDATE = False

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
train_df = pd.read_csv(path+"train.csv", skiprows=range(1,94903891), nrows=90000000,dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
print('load test...')
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

import gc

len_train = len(train_df)
train_df=train_df.append(test_df)

del test_df
gc.collect()

print('data prep...')

# Fix stuff that got float64-ified because of missing values when train and test were concatenated
train_df['is_attributed'] = train_df['is_attributed'].fillna(value=99).astype('uint8')
train_df['click_id'] = train_df['click_id'].fillna(value=-1).astype('int32')

train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df.drop(['click_time'], axis=1, inplace=True)

gc.collect()


#----------------------------------------------------------------
print('group by : ip_app_channel_var_day')
gp = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
train_df['ip_app_channel_var_day'] = train_df['ip_app_channel_var_day'].astype('float32')
gc.collect()
#-------------------------------------------------------------------------------

print('group by : ip_day_hour_count_chl')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
del gp
train_df['qty'] = train_df['qty'].astype('uint16')
gc.collect()

print('group by : ip_app_count_chl')
gp = train_df[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp, on=['ip','app'], how='left')
del gp
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
gc.collect()


print('group by : ip_app_os_count_chl')
gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')
gc.collect()
#-------------------------------------------------------------------------------

print('group by : ip_day_chl_var_hour')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'qty_var'})
train_df = train_df.merge(gp, on=['ip','day','channel'], how='left')
del gp
train_df['qty_var'] = train_df['qty_var'].astype('float32')
gc.collect()


print('group by : ip_app_os_var_hour')
gp = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
train_df['ip_app_os_var'] = train_df['ip_app_os_var'].astype('float32')
gc.collect()
#-------------------------------------------------------------------------------

print('group by : ip_app_chl_mean_hour')
gp = train_df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
train_df['ip_app_channel_mean_hour'] = train_df['ip_app_channel_mean_hour'].astype('float32')
gc.collect()

#---------------------------------------------------------------------------------
print("vars and data type: ")
train_df.info()

metrics = 'auc'
lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':metrics,
        'learning_rate': 0.1,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 13,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 4,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.89,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
        'scale_pos_weight':77, # because training data is extremely unbalanced 
        'metric':metrics
}

target = 'is_attributed'
predictors = ['app','device','os', 'channel', 'hour', 'day', 'qty', 'ip_app_count', 'ip_app_os_count','qty_var', 'ip_app_os_var','ip_app_channel_var_day','ip_app_channel_mean_hour']
categorical = ['app','device','os', 'channel', 'hour']

print(train_df.head(5))

if VALIDATE:

    test_df = train_df[len_train:]
    val_df = train_df[(len_train-3000000):len_train]
    train_df = train_df[:(len_train-3000000)]
    test_df.drop(['is_attributed'], axis=1, inplace=True)
    train_df.drop(['click_id'], axis=1, inplace=True)
    val_df.drop(['click_id'], axis=1, inplace=True)

    print(train_df.info())
    print(val_df.info())
    print(test_df.info())

    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))
    print("test size : ", len(test_df))


    gc.collect()

    print("Training...")

    num_boost_round=1000
    early_stopping_rounds=100

    xgtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
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
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=None)

    n_estimators = bst.best_iteration

    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])
    
else:

    test_df = train_df[len_train:]
    train_df = train_df[:len_train]
    test_df.drop(['is_attributed'], axis=1, inplace=True)
    train_df.drop(['click_id'], axis=1, inplace=True)

    print(train_df.info())
    print(test_df.info())

    print("train size: ", len(train_df))
    print("test size : ", len(test_df))

    gc.collect()

    print("Training...")

    num_boost_round=500

    xgtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del train_df
    gc.collect()

    evals_results = {}

    bst = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain], 
                     valid_names=['train'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     verbose_eval=10, 
                     feval=None)

    n_estimators = num_boost_round

    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['train'][metrics][n_estimators-1])
    

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id']

print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])
print("writing...")
sub.to_csv('sub_lgb_balanced55.csv',index=False)
print("done...")
print(sub.info())
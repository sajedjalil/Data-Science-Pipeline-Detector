"""
Original kernel:non-blending lightGBM model LB: 0.977:https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977?scriptVersionId=3224614
V0 Modified by Andy:Kaggle-runnable version of Baris Kanber's LightGBM:https://www.kaggle.com/aharless/kaggle-runnable-version-of-baris-kanber-s-lightgbm/comments
A non-blending lightGBM model that incorporates portions and ideas from various public kernels.

Modified by Hugo:
- Take a random subsample with 20kk entries from train dataset
- Incorporate new features to be used in the model
- Delete/Comment features that were less significant to the problem evaluated
- Added new fuctions
- Added comments to all functions

My References:
- https://www.kaggle.com/wenjiebai/if-you-run-on-entire-dataset-lb-0-9798 (Original Script that I've forked)
- https://www.kaggle.com/yuliagm/how-to-work-with-big-datasets-on-16g-ram-dask
- https://www.kaggle.com/nanomathias/feature-engineering-importance-testing
- https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977
- https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-lb-0-9680
- https://www.kaggle.com/asraful70/talkingdata-added-new-features-in-lightgbm
- https://www.kaggle.com/joaopmpeinado/talkingdata-xgboost-lb-0-966

"""
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os

predictors = []

def skiplines_train(remaining_Lines):
    '''
    Choosing random entries to be ignored when importing a dataset
    
    remaining_Lines (int): how many data entries will be imported from original dataset. They are randomly choosen
    
    Contribuiton from: https://www.kaggle.com/yuliagm/how-to-work-with-big-datasets-on-16g-ram-dask
    '''
    lines = 184903890    #already disconsidered the head line
    
    #generate list of lines to skip
    skiplines = np.random.choice(np.arange(1, lines+1), size=lines-remaining_Lines, replace=False)

    #sort the list
    return np.sort(skiplines)

def do_count( df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    '''
    Does a groupby on columns 'group_cols' and count how many entries on each group. Transform into a dataframe and merge with the original one, adding a new feature.
    
    Append the new feauture to predictors list
    '''
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left', copy=False)
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type,copy=False)
    predictors.append(agg_name)
    return( df )

def do_countuniq( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    '''
    Does a groupby on columns 'group_cols' and count how many unique 'counted' feature entries on each group. Transform into a dataframe and merge with the original one, adding a new feature.
    
    Append the new feauture to predictors list
    '''
    if show_agg:
        print( "Counting unqiue ", counted, " by ", group_cols , '...' )
    # print('the Id of train_df while function before merge: ',id(df)) # the same with train_df
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left', copy=False)
    # print('the Id of train_df while function after merge: ',id(df)) # id changes
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type,copy=False)
    predictors.append(agg_name)
    return( df )
    
def do_cumcount( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    '''
    Does a groupby on columns 'group_cols' and cumulative count of 'counted' feature on each group. Transform into a dataframe and merge with the original one, adding a new feature.
    
    Append the new feauture to predictors list
    '''
    
    if show_agg:
        print( "Cumulative count by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type,copy=False)
    predictors.append(agg_name)
    return( df )

def do_mean( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    '''
    Does a groupby on columns 'group_cols' and find the mean of 'counted' feature on each group. Transform into a dataframe and merge with the original one, adding a new feature.
    
    Append the new feauture to predictors list
    '''
    
    if show_agg:
        print( "Calculating mean of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left', copy=False)
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type,copy=False)
    predictors.append(agg_name)
    return( df )

def do_var( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    '''
    Does a groupby on columns 'group_cols' and find the variance of 'counted' feature on each group. Transform into a dataframe and merge with the original one, adding a new feature.
    
    Append the new feauture to predictors list
    '''
    
    if show_agg:
        print( "Calculating variance of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left', copy=False)
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type,copy=False)
    predictors.append(agg_name)
    return( df )
    
def do_next_Click( df,agg_suffix='nextClick_seconds', agg_type=np.float32):
    '''
    Take a list of groupby columns. For each loop, generate new feature indicating how many seconds for the next click with equal features based on groupby columns, 
    retunrs NaN if there is no equivalent click after the entry evaluated.
    
    Add the new feature to original dataframe. Append the new feauture to predictors list.
    '''
    print(f">> \nExtracting {agg_suffix} time calculation features...\n")
    
    GROUP_BY_NEXT_CLICKS = [
    {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    {'groupby': ['ip', 'app']},
    {'groupby': ['ip', 'os', 'device']},
    {'groupby': ['ip', 'os', 'device', 'app']},
    {'groupby': ['device']},
    {'groupby': ['device', 'channel']},     
    {'groupby': ['app', 'device', 'channel']},
    {'groupby': ['device', 'hour']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']
    
    
        # Run calculation
        print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        df[new_feature] = (df[all_features].groupby(spec['groupby']).click_time.shift(-1) - df.click_time).astype(agg_type)
        
        predictors.append(new_feature)
        gc.collect()
        
    return (df)
    
def do_prev_Clicks_count( df,agg_suffix='prev_Clicks_count', agg_type='uint32'):
    '''
    Take a list of groupby columns. For each loop, generate new feature indicating how many previous clicks have equal features based on groupby columns, 
    retunrs 0 if there is no equivalent click prior to the entry evaluated.
    
    Add the new feature to original dataframe. Append the new feauture to predictors list.
    '''
    
    print(f">> \nExtracting {agg_suffix} features...\n")
    
    HISTORY_CLICKS = [
    {'agg_suffix' : 'identical_clicks', 'groupby' : ['ip', 'app', 'device', 'os', 'channel']},
    {'agg_suffix' : 'app_clicks',  'groupby' : ['ip', 'app']},
    {'agg_suffix' : 'app_channel_clicks',  'groupby' : ['ip', 'app', 'channel']},
    {'agg_suffix' : 'app_device_clicks',  'groupby' : ['ip', 'app', 'device']}
    ]

    # Calculate the time to next click for each group
    for spec in HISTORY_CLICKS:
    
        # Name of new feature
        new_feature = 'prev_{}'.format(spec['agg_suffix'])    
    
        # Unique list of features to select
        all_features = spec['groupby']
        
        # Run calculation
        print(f">> Grouping by {spec['groupby']}, and saving count in: {new_feature}")
        df[new_feature] = (df[all_features].groupby(spec['groupby']).cumcount()).astype(agg_type, copy=False)
        
        predictors.append(new_feature)
        gc.collect()
        
    return (df)

def do_prev_Click( df,agg_suffix='prevClick_seconds', agg_type=np.float32):
    '''
    Take a list of groupby columns. For each loop, generate new feature indicating how many seconds from previous click with equal features based on groupby columns, 
    retunrs NaN if there is no equivalent click prior to the entry evaluated.
    
    Add the new feature to original dataframe. Append the new feauture to predictors list.
    '''
    
    #Previous clicks is very correlated to next clicks, so did not need to take many different groups.
    
    print(f">> \nExtracting {agg_suffix} time calculation features...\n")
    
    GROUP_BY_NEXT_CLICKS = [
    
    # V1
    # {'groupby': ['ip']},
    # {'groupby': ['ip', 'app']},
    {'groupby': ['ip', 'channel']},
    # {'groupby': ['ip', 'os']},
    
    # V3
    #{'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    #{'groupby': ['ip', 'os', 'device']},
    #{'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        df[new_feature] = (df.click_time - df[all_features].groupby(spec['groupby']).click_time.shift(+1) ).astype(agg_type)
        
        predictors.append(new_feature)
        gc.collect()
    return (df)

def do_confRate(df, train_len, agg_type='float32'):
    '''
    Take a list of groupby columns. For each loop, generate new feature indicating Confidence Rates for is_attributed based on groupby columns.
    Only evaluated on train data entries.
    
    Add the new feature to original dataframe. Append the new feauture to predictors list.
    
    Reference: https://www.kaggle.com/nanomathias/feature-engineering-importance-testing
    '''
    
    ATTRIBUTION_CATEGORIES = [        
        # V1 Features #
        ['ip'], ['app'], ['device'], ['os'], ['channel'],
        
        # V2 Features #
        ['app', 'channel'],
        ['app', 'os'],
        ['app', 'device'],
        
        # V3 Features #
        ['channel', 'os'],
        ['channel', 'device'],
        ['os', 'device']
    ]
    
    # Find frequency of is_attributed for each unique value in column
    for cols in ATTRIBUTION_CATEGORIES:
        
        # New feature name
        new_feature = '_'.join(cols)+'_confRate'    
        
        # Unique list of features to select
        all_features = cols + ['is_attributed']
        
        # Perform the groupby
        group_object = df[all_features].iloc[:train_len].groupby(cols)
        
        # Group sizes    
        group_sizes = group_object.size()
        log_group = np.log(100000) # 1000 views -> 60% confidence, 100 views -> 40% confidence 
        
        # Aggregation function
        def rate_calculation(x):
            """Calculate the attributed rate. Scale by confidence"""
            rate = x.sum() / float(x.count())
            conf = np.min([1, np.log(x.count()) / log_group])
            return rate * conf
        
        # Perform the merge
        print(f">> Grouping by {cols}, and saving confRate in: {new_feature} ...")
        gp = group_object['is_attributed'].apply(rate_calculation).reset_index().rename(index=str, columns={'is_attributed': new_feature})
        df = df.merge(gp, on=cols, how='left', copy=False)
        
        del gp
        gc.collect()
        
        df[new_feature] = df[new_feature].astype(agg_type,copy=False)
        predictors.append(new_feature)

    return (df)

def lgb_modelfit_nocv(dtrain, dvalid, predictors, target='target', feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None,metrics='auc'):
    '''
    Train Light GB Model, returns best fitted over validation dataset.
    '''
    
    # Parameters taken from others Kernels. I did not fined tuned them
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'auc',
        'learning_rate': 0.2, # 【consider using 0.1】
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'scale_pos_weight': 200, # because training data is extremely unbalanced
        'num_leaves': 7,  # we should let it be smaller than 2^(max_depth), default=31
        'max_depth': 3,  # -1 means no limit, default=-1
        'min_data_per_leaf': 100,  # alias=min_data_per_leaf , min_data, min_child_samples, default=20
        'max_bin': 100,  # Number of bucketed bin for feature values,default=255
        'subsample': 0.7,  # Subsample ratio of the training instance.default=1.0, alias=bagging_fraction
        'subsample_freq': 1,  # k means will perform bagging at every k iteration, <=0 means no enable,alias=bagging_freq,default=0
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.alias:feature_fraction
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf),default=1e-3,Like min_data_in_leaf, it can be used to deal with over-fitting
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4, # should be equal to REAL cores:http://xgboost.readthedocs.io/en/latest/how_to/external_memory.html
        'verbose': 0
    }
    
    #lgb_params.update(params) # Python dict.update()

    print("load train_df into lgb.Dataset...")
    # free_raw_data (bool, optional (default=True)) – If True, raw data is freed after constructing inner Dataset.
    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features)
    
    del dtrain
    print("load valid_df into lgb.Dataset...")
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features)
    
    del dvalid
    gc.collect()

    evals_results = {}
    
    # Warning:basic.py:681: UserWarning: categorical_feature in param dict is overrided.
    # https://github.com/Microsoft/LightGBM/blob/master/python-package/lightgbm/basic.py#L679
    # https://github.com/Microsoft/LightGBM/blob/master/python-package/lightgbm/basic.py#L483
    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgvalid], 
                     valid_names=['valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)
    
    del xgtrain, xgvalid
    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst1.best_iteration-1])
    gc.collect()

    return (bst1,bst1.best_iteration)
    
def lgb_modelfit_nocv_split(train_x, val_x, train_y, val_y, predictors, feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None,metrics='auc'):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'auc',
        'learning_rate': 0.2, # 【consider using 0.1】
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'scale_pos_weight': 200, # because training data is extremely unbalanced
        'num_leaves': 7,  # we should let it be smaller than 2^(max_depth), default=31
        'max_depth': 3,  # -1 means no limit, default=-1
        'min_data_per_leaf': 100,  # alias=min_data_per_leaf , min_data, min_child_samples, default=20
        'max_bin': 100,  # Number of bucketed bin for feature values,default=255
        'subsample': 0.7,  # Subsample ratio of the training instance.default=1.0, alias=bagging_fraction
        'subsample_freq': 1,  # k means will perform bagging at every k iteration, <=0 means no enable,alias=bagging_freq,default=0
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.alias:feature_fraction
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf),default=1e-3,Like min_data_in_leaf, it can be used to deal with over-fitting
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4, # should be equal to REAL cores:http://xgboost.readthedocs.io/en/latest/how_to/external_memory.html
        'verbose': 0
    }
    
    #lgb_params.update(params) # Python dict.update()

    print("load train_df into lgb.Dataset...")
    # free_raw_data (bool, optional (default=True)) – If True, raw data is freed after constructing inner Dataset.
    xgtrain = lgb.Dataset(train_x.values, label=train_y.values,
                          feature_name=predictors,
                          categorical_feature=categorical_features)
    
    del train_x
    del train_y
    
    print("load valid_df into lgb.Dataset...")
    xgvalid = lgb.Dataset(val_x.values, label=val_y.values,
                          feature_name=predictors,
                          categorical_feature=categorical_features)
    
    del val_x
    del val_y
    gc.collect()

    evals_results = {}
    
    # Warning:basic.py:681: UserWarning: categorical_feature in param dict is overrided.
    # https://github.com/Microsoft/LightGBM/blob/master/python-package/lightgbm/basic.py#L679
    # https://github.com/Microsoft/LightGBM/blob/master/python-package/lightgbm/basic.py#L483
    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgvalid], 
                     valid_names=['valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)
    
    del xgtrain, xgvalid
    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst1.best_iteration-1])
    gc.collect()

    return (bst1,bst1.best_iteration)

# --------------------------------------------------------------------------------------------------------------
def DO(frm,to,fileno):
    '''
    Import train and test dataset. Create new features, train the model with train dataset, predict test target values. Also save image with features importance.
    '''
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32', 
            }
    
    print('loading train data...',frm,to)
    # usecols:Using this parameter results in much faster parsing time and lower memory usage.
    train_df = pd.read_csv('../input/generate-train-20kk-random-lines/train_20kk_Lines.csv', parse_dates=['click_time'], skiprows=range(1,frm), nrows=to-frm, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
       
    print('loading test data...')
    if debug:
        test_df = pd.read_csv("../input/talkingdata-adtracking-fraud-detection/test.csv", nrows=1000000, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    else:
        test_df = pd.read_csv("../input/talkingdata-adtracking-fraud-detection/test.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

    len_train = len(train_df)
    train_df=train_df.append(test_df) # Shouldn't process individually,because of lots of count,mean,var variables
    
    train_df['is_attributed'].fillna(-1,inplace=True)
    train_df['is_attributed'] = train_df['is_attributed'].astype('uint8',copy=False)
    
    train_df['click_id'].fillna(-1,inplace=True)
    train_df['click_id'] = train_df['click_id'].astype('uint32',copy=False)
    
    del test_df
    gc.collect()
    
    print('Extracting new features...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    gc.collect()
    
    
    # Commented this feature out because it reduces public score from: 0.9762931 -> to: 0.8729905
    #train_df = do_confRate(train_df, len_train)                     #New
    
    #Assuming IPs could represent a single user or household, or it could be a public network with thousands different devices accessing different apps and click on ads through different channels
    #I'd like to create new features to identify some user/network patterns to identify fraudulent clicks
    
    #How many unique channels are used from the same ip?
    train_df = do_countuniq( train_df, ['ip'], 'channel', 'ip_unique_channel', 'uint8', show_max=False ); gc.collect()                              #ip                 unique      channel
    #Are there users clicking on ads from this IP 24h each day? Is there any hour that people do not click on ads through this IP?
    train_df = do_countuniq( train_df, ['ip', 'day'], 'hour', 'ip_day_unique_hour', 'uint8', show_max=False ); gc.collect()                         #ip_day             unique      hour
    #How many different apps are accessed from each IP?
    train_df = do_countuniq( train_df, ['ip'], 'app', 'ip_unique_app', 'uint16', show_max=False ); gc.collect()                                     #ip                 unique      app
    #How many different types of OS click on adds in the same IP and App?
    train_df = do_countuniq( train_df, ['ip', 'app'], 'os', 'ip_app_unique_os', 'uint8', show_max=False ); gc.collect()                             #ip_app             unique      os
    #How many different device types are click on adds through the same ip?
    train_df = do_countuniq( train_df, ['ip'], 'device', 'ip_unique_device', 'uint16', show_max=False ); gc.collect()                               #ip                 unique      device
    #Clicks with the same IP, Device and OS probably is the same user/fraudster. How many different apps is each user accessing?
    train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app', 'ip_device_os_unique_app','uint8', show_max=False ); gc.collect()             #ip_device_os       unique      app
    #How many different channels are used for each app?
    train_df = do_countuniq( train_df, ['app'], 'channel', 'app_unique_channel','uint8', show_max=False ); gc.collect()                             #app                unique      channel
    #How many different apps access each channel?
    train_df = do_countuniq( train_df, ['channel'], 'app', 'channel_unique_app','uint8', show_max=False ); gc.collect()                             #channel            unique      app             #New
    
    #Considering that the data entries are ordered by click_time, each function below finds how many times each counted feature had appeared
    #before on clicks that share the same groupby features.
    
    #How many times each OS appear on the same ip over the train and test period
    train_df = do_cumcount( train_df, ['ip'], 'os', 'ip_cumcount_os', show_max=False ); gc.collect()                                                #ip                 cumcount    os
    #How many each channel is accessed by the same ip on each hour of each day
    train_df = do_cumcount( train_df, ['ip', 'day', 'hour'], 'channel', 'ip_day_hour_cumcount_channel','uint16',show_max=False ); gc.collect()      #ip_day_hour        cumcount    channel         #New
    #How many times the user/fraudster accessed the same app. Is the fraudster targetting a single app or many?
    train_df = do_cumcount( train_df, ['ip', 'device', 'os'], 'app', 'ip_device_os_cumcount_app', show_max=False ); gc.collect()                    #ip_device_os       cumcount    app
    #How many times each app are accessed from the same ip
    train_df = do_cumcount( train_df, ['ip'], 'app', 'ip_cumcount_app', show_max=False ); gc.collect()                                              #ip                 cumcount    app             #New
    
    #Count how many clicks occur from each ip over each hour and day
    train_df = do_count( train_df, ['ip', 'day', 'hour'], 'ip_tcount','uint16',show_max=False ); gc.collect()                                       #ip time            count
    #Count how many clicks occur from each ip over each hour, aggregating days
    train_df = do_count( train_df, ['ip', 'hour'], 'ip_tcount2','uint32',show_max=False ); gc.collect()
    #Count how many clicks occur from each ip on each app
    train_df = do_count( train_df, ['ip', 'app'], 'ip_app_count','uint32', show_max=False ); gc.collect()                                           #ip_app             count
    #Count how many clicks occur from each ip and OS on each app
    train_df = do_count( train_df, ['ip', 'app', 'os'], 'ip_app_os_count', 'uint16', show_max=False ); gc.collect()                                 #ip_app_os          count
    
    
    train_df = do_var( train_df, ['ip', 'day', 'channel'], 'hour', 'ip_tchan_count', show_max=False ); gc.collect()                                 #ip_day_channel     var         hour
    train_df = do_var( train_df, ['ip', 'app', 'os'], 'hour', 'ip_app_os_var', show_max=False ); gc.collect()                                       #ip_app_os          var         hour
    train_df = do_var( train_df, ['ip', 'app', 'channel'], 'day', 'ip_app_channel_var_day', show_max=False ); gc.collect()                          #ip_app_channel     var         day
    
    train_df = do_mean( train_df, ['ip', 'app', 'channel'], 'hour', 'ip_app_channel_mean_hour', show_max=False ); gc.collect()                      #ip_app_channel     mean        hour


    print('doing nextClick & prev_Click...')
    
    train_df['click_time'] = (train_df['click_time'].astype(np.int64,copy=False) // 10 ** 9).astype(np.int32,copy=False)
    
    
    train_df = do_prev_Clicks_count(train_df); gc.collect()         #New
    train_df = do_next_Click(train_df); gc.collect()                #New
    train_df = do_prev_Click(train_df); gc.collect()                #New
    
    print('nextClick & prev_Click are Done')
   
   
    
    train_df.drop(['click_time','day'], axis=1, inplace=True)
    gc.collect()
    
#----------------------------------------------------------------------------------------------------------------
    print("vars and data type: ")
    target = 'is_attributed'
    predictors.extend(['app','device','os', 'channel', 'hour'])
    categorical = ['app', 'device', 'os', 'channel', 'hour',]
    print('predictors',predictors)

    test_df = train_df[len_train:]
    test_df.drop(columns='is_attributed',inplace=True)
    train_df.drop(columns='click_id',inplace=True)
    train_df = train_df[:len_train]
    
    train_x, val_x, train_y, val_y = train_test_split(train_df[predictors], train_df[target], test_size=0.3, random_state=0)
    
    print("train size: ", len(train_df))
    print("test size : ", len(test_df))
    
    del train_df
    gc.collect()
    
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id']
    gc.collect()

    print("Training...")
    start_time = time.time()
    
    
    # Public AUC sem train_test_split: 0.9570213 / Public AUC com train_test_split: 0.9570636
    # Ambos rodados com as features iniciais, sem nenhuma adição de feature para conseguir rodar no próprio kernel do kaggle.
    # Apesar de não conseguir comparar os resultados com todas as features adicionadas, optou-se por deixar o script com train_test_split a partir do resultado parcial obtido.
    
    (bst,best_iteration) = lgb_modelfit_nocv_split(train_x, val_x, train_y, val_y, predictors, early_stopping_rounds=50, verbose_eval=True, num_boost_round=2000, categorical_features=categorical)

    del train_x
    del val_x
    del train_y
    del val_y
    gc.collect()
    
    print('[{}]: model training time'.format(time.time() - start_time))


    print('Plot feature importances...')
    lgb.plot_importance(bst)
    # plt.show()
    plt.gcf().savefig('feature_importance_runnablelightgbm_split.png')
    lgb.plot_importance(bst,importance_type='gain')
    plt.gcf().savefig('feature_importance_runnablelightgbm_gain.png')

    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors],num_iteration=best_iteration)
    del test_df
    if not debug:
        print("writing...")
        sub.to_csv('sub_it%d.csv'%(fileno),index=False,float_format='%.9f')
    del sub
    gc.collect()
    print("All done...")
    

# Main function-------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    inpath = '../input/generate-train-20kk-random-lines'
   
    nrows=20000000 # the first line is columns' name
    nchunk=nrows # 【The more the better】
    val_size=2500000
    frm=0
    
    debug=False
    #debug=True
    if debug:
        print('*** Debug: this is a test run for debugging purposes ***')
        frm=0
        nchunk=1000000
        val_size=100000
    
    to=frm+nchunk
    
    DO(frm,to,0) # fileno start from 0
    
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt

# Preprocessing function

# cleansing function

def preprocessing(chunk):
    
    print('preprocessing 1st batch of features...')
    chunk['hour'] = pd.to_datetime(chunk.click_time).dt.hour.astype('uint8')
    chunk['day'] = pd.to_datetime(chunk.click_time).dt.day.astype('uint8')
    chunk['second'] = pd.to_datetime(chunk.click_time).dt.second.astype('uint8')
    chunk['click_time'] = (chunk['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
    chunk['nextClick'] = (chunk.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1) - chunk.click_time).astype(np.float32)
    
    # # of clicks for each ip-day-hour combination
    print('preprocessing 2nd batch of features...')
    gp = chunk[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
    chunk = chunk.merge(gp, on=['ip','day','hour'], how='left')
    del gp
    gc.collect()

    # # of clicks for each ip-app combination    
    gp = chunk[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    chunk = chunk.merge(gp, on=['ip','app'], how='left')
    del gp
    gc.collect()

    # # of clicks for each ip-app-os combination
    gp = chunk[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    chunk = chunk.merge(gp, on=['ip','app', 'os'], how='left')
    del gp
    gc.collect()
    
    del chunk['day']
    del chunk['second']
    
    
    gc.collect()
    
    # NOT ENOUGH MEMORY
    # Historical Clicks
    print('Preprocessing 3rd batch of features...')
    HISTORY_CLICKS = {
    'identical_clicks': ['ip', 'app', 'device', 'os', 'channel'],
    'app_clicks': ['ip', 'app']
    }

    # Go through different group-by combinations
    for fname, fset in HISTORY_CLICKS.items():

        # Clicks in the past
        chunk['prev_'+fname] = chunk. \
            groupby(fset). \
            cumcount(). \
            rename('prev_'+fname)

        # Clicks in the future
        chunk['future_'+fname] = chunk.iloc[::-1]. \
            groupby(fset). \
            cumcount(). \
            rename('future_'+fname).iloc[::-1]
    
    # Does not improve AUC
    #chunk['prevClick'] = (chunk.click_time - chunk.groupby(['ip', 'app', 'device', 'os']).click_time.shift(+1)).astype(np.float32)
    del chunk['ip']
    gc.collect()
    
    print('Done !')
    
    return chunk
    
    

def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None, init_model=None):
                     
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.01,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
        'metric':metrics
    }
    
    lgb_params.update(params)
    
    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval,
                     init_model=init_model)

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    return bst1


def main():
    
    # consider removing weak predictors such as second and day
    
    predictors = ['app', 'device', 'os', 'channel', 'hour', 'nextClick', 'qty',
           'ip_app_count', 'ip_app_os_count']
    categorical = ['app','device','os', 'channel', 'hour']
    target = 'is_attributed'
    
    params = {
    'learning_rate': 0.1,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':99 # because training data is extremely unbalanced 
    }
    
    # USER DEFINED
    chunksize = 40000000
    chunk_idx = 0
    chunk_max = 1
    split_ratio = 0.9
    
    
    for chunk in pd.read_csv('../input/train.csv', chunksize=chunksize, parse_dates = ['click_time']):
        # Extracting new features
        chunk = preprocessing(chunk)
        val_df = chunk[int(split_ratio*chunksize):chunksize]
        #chunk = chunk[:int(split_ratio*chunksize)]


        if chunk_idx > 0: # not load in first run
            bst = lgb_modelfit_nocv(params,
                        chunk, 
                        val_df, 
                        predictors, 
                        target, 
                        objective='binary', 
                        metrics='auc',
                        early_stopping_rounds=50, 
                        verbose_eval=True, 
                        num_boost_round=500, 
                        categorical_features=categorical, init_model=bst)
        else:
            bst = lgb_modelfit_nocv(params,
                        chunk, 
                        val_df, 
                        predictors, 
                        target, 
                        objective='binary', 
                        metrics='auc',
                        early_stopping_rounds=50, 
                        verbose_eval=True, 
                        num_boost_round=500, 
                        categorical_features=categorical)
        print("Processed chunk %d" % chunk_idx)
        chunk_idx = chunk_idx + 1
        if chunk_idx >= chunk_max:
            break
    
    test_df = pd.read_csv('../input/test.csv', parse_dates= ['click_time'])
    # current after preprocessing
    test_df =preprocessing(test_df)

    del val_df
    gc.collect()
    
    
    ax = lgb.plot_importance(bst, max_num_features=100)
    plt.show()
    plt.savefig('foo.png')
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')
    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors])
    print("writing...")
    sub.to_csv('sub_lgb_balanced99.csv',index=False)
    print("done...")
    
if __name__ == '__main__':
    main()
    
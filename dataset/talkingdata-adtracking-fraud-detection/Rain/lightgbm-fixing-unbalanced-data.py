"""
This version has improvements based on new feature engg techniques observed from different kernels. Below are few of them:

"""

import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc

def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.1,
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
        'nthread': 4,
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
                     )

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    return bst1

path = '../input/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        #'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }



"""
 read kernel data
"""
train_df = pd.read_pickle('../input/training-and-validation-data-pickle/training.pkl.gz')
valid_df = pd.read_pickle('../input/training-and-validation-data-pickle/validation.pkl.gz')
test_df = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/test.csv', dtype=dtypes, usecols=['ip','app','device','os', 'channel','click_id','click_time'])
sub = pd.DataFrame()
sub['click_id'] = test_df['click_id']
test_df = test_df.drop('click_id',1)
train_df = train_df.append(valid_df)
train_df = train_df.append(test_df)
train_df = pd.concat([pd.read_pickle('../input/feature1/fea1.pkl.gz'),train_df],axis=1)
train_df['hour']=pd.to_datetime(train_df['click_time']).dt.hour.astype(np.uint8)
train_df['day']=pd.to_datetime(train_df['click_time']).dt.day.astype(np.uint8)
del train_df['click_time']


test_df = train_df[122071523+20898422:]
val_df = train_df[122071523:122071523+20898422]
train_df = train_df[:122071523]

#down-sample
train_df=pd.concat([train_df[train_df['is_attributed']==0].sample(frac=0.1),train_df[train_df['is_attributed']==1]]).reset_index(drop=True)

"""
k-fold cv
"""
#from sklearn.model_selection import train_test_split
#train_df,val_df = train_test_split(train,test_size = 0.2,random_state = 0) 




print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))

target = 'is_attributed'
predictors = list(train_df.drop(['ip','is_attributed'],1).columns)
print(predictors)
categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']

gc.collect()

print("Training...")
start_time = time.time()


params = {
    'learning_rate': 0.15,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 7,  # 2^max_depth - 1
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':200 # because training data is extremely unbalanced 
}
bst = lgb_modelfit_nocv(params, 
                        train_df, 
                        val_df, 
                        predictors, 
                        target, 
                        objective='binary', 
                        metrics='auc',
                        early_stopping_rounds=20, 
                        verbose_eval=True, 
                        num_boost_round=300, 
                        categorical_features=categorical)

print('[{}]: model training time'.format(time.time() - start_time))
del train_df
del val_df
gc.collect()

print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])
print("writing...")
sub.to_csv('sub_lgb_balanced99.csv',index=False)
print("done...")
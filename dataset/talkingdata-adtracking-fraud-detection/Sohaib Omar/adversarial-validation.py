"""
Attempt to find out if train and test set differ using Adversarial Validation approach. This kernel is inspired from below kernels
- https://www.kaggle.com/ogrellier/adversarial-validation-and-lb-shakeup
- https://www.kaggle.com/konradb/adversarial-validation-and-other-scary-terms

Whats Adversarial Validation?
    Adversarial validation is a mean to check if train and test datasets have significant differences. 
    The idea is to use the dataset features to try and separate train and test samples.

    So you would create a binary target that would be 1 for train samples and 0 for test samples and 
    fit a classifier on the features to predict if a given sample is in train or test datasets!
    
    If validation scores of folds are indistinguishable then we can agree that train and test set are similar but if 
    validation score accross folds is distinguished easily then train and test set distribution may differ.
    
    To read more about Adversarial validation visit below post:
    - http://fastml.com/adversarial-validation-part-two/
"""

import pandas as pd 
import numpy as np
import time
import gc

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from datetime import datetime


#############################################################################
#define important variables
TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'

##LGB params
BOOST_ROUNDS = 350
EARLY_STOPPING = 30


categorical = ['app','device',  'os', 'channel']
predictors = ['app','device',  'os', 'channel']
SKIP = range(1,144903891)
NROWS=40000000
RANDOM_SEED = 1001
NUM_FOLDS = 5
train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }


lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'auc',
        'learning_rate': 0.05,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 35,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 4,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.8,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 7,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'nthread': 4,
        'verbose': -1  }
        
###################################
#helper functions

def timer(start_time=None):
    """Prints time
    __author__ = @Tilii
    
    Initiate a time object, and prints total time consumed when again initialized object is passed as argument
    
    Keyword Arguments:
        start_time {[object]} -- initialized time object (default: {None})
    
    """
    
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

#############################################################################MAIN########################################

t = timer(None)
print("Loading train and test set")
train = pd.read_csv(TRAIN_PATH,skiprows=SKIP, header=0,nrows=NROWS, usecols=train_columns, dtype=dtypes)
test = pd.read_csv(TEST_PATH, dtype=dtypes, usecols=test_columns)
timer(t)

train.drop(['is_attributed', 'click_time'], axis=1, inplace=True)
test.drop(['click_id', 'click_time'], axis=1, inplace=True)

#assign target if set is test set or not
train['is_test'] = 0
test['is_test'] = 1

adv_set = pd.concat([train, test])

del train, test

target = adv_set.is_test.values
adv_set.drop(['is_test'], axis=1, inplace=True)

train = adv_set[predictors].values

fold_scores = []

print("Starting adversarial validation")
folds = list(StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(train, target))

for i, (train_idx, val_idx) in enumerate(folds):
    print("Seperating Val and train set")
    X_train = train[train_idx]
    y_train = target[train_idx]
    
    X_val = train[val_idx]
    y_val = target[val_idx]
    gc.collect()
    #make lgb train and val
    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=predictors, categorical_feature=categorical)
    lgb_val = lgb.Dataset(X_val, label=y_val, feature_name=predictors, categorical_feature=categorical)
    
    gc.collect()
    
    print("Training Lgb")
    #run lgb
    lgb_model = lgb.train(lgb_params, lgb_train,valid_sets=[lgb_train, lgb_val], valid_names=['train', 'val'], num_boost_round=BOOST_ROUNDS, verbose_eval=5, early_stopping_rounds=EARLY_STOPPING)
    fold_scores.append(lgb_model.best_score)
    # print(lgb_model.best_score)


for i, score in enumerate(fold_scores):
    print("Fold {} Val Auc:{:.6f}".format(str(i), score['val']['auc']))

print("\n")
for i, score in enumerate(fold_scores):
    print("Fold {} Train Auc:{:.6f}".format(str(i), score['train']['auc']))
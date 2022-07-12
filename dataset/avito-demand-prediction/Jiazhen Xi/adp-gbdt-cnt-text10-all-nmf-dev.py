import warnings
warnings.filterwarnings("ignore")
import os
import gc
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
os.environ['OMP_NUM_THREADS'] = '4'

import sys
UTILS_PATH = '../input/myutils/'
if UTILS_PATH not in sys.path:
    sys.path.append(UTILS_PATH)

import adp_utils as au
    
if __name__ == '__main__':
    eval_sets, y, train_num = au.load_eval_sets_lables()
    
    valid_fold = 0
    SEED = 1515333 + valid_fold
    X = [
        au.get_common_feats(),
        #au.get_lbe_feats(),
        au.get_enc_feats('count', enc_flags=[1, 1, 1]),
        au.get_nmf_feats(),
    ]
    X = au.merge_dfs(X)
    print('df total usage:\n')
    X.info(False)
    feature_name = au.get_feature_name(X)
    categorical = au.get_cat_cols(X)
    
    X, X_test = au.get_test_split(X, train_num)
    X_train, X_valid, y_train, y_valid = au.get_train_valid_split(
        X, y, valid_fold, eval_sets
    )
    lgb_params = {}
    au.train_lgb(
        X_train, y_train, X_valid, y_valid, X_test, 
        valid_fold, 
        feature_name, categorical, 
        SEED=SEED, lgb_params=lgb_params, 
        num_boost_round=20000, early_stopping_rounds=200, #200
        verbose_eval=25
    )
    
    
    
    
    
    
    
    
    
    
    
    
    
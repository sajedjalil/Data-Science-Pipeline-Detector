# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import warnings
import time
warnings.filterwarnings("ignore")
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score
import gc

#fucntion for bayesian optimization on lightGBM
def bayes_parameter_opt_lgbm(X, y, init_round = 15, opt_round = 25, n_folds = 5, random_seed = 6, n_estimators = 10000, 
learning_rate = 0.05, output_process=False, early_stopping_round = 1000, verbose_eval = 2500):
    train_data = lgb.Dataset(data = X, label = y)
    
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, lambda_l1, lambda_l2, min_split_gain,min_child_weight,
    min_data_in_leaf,min_sum_heassian_in_leaf):
        params = {'application':'binary','num_iterations': n_estimators, 'learning_rate':learning_rate, 
        'early_stopping_round': early_stopping_round, 'metric':'auc', 'max_depth':-1}
        params["num_leaves"] = int(round(num_leaves)) #since num_leaves should be a integer so we are converting it to integer
        params['feature_fraction'] = max(min(feature_fraction, 1), 0) # it should be between 0 to 1
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0) # it should be between 0 to 1
        params['lambda_l1'] = max(lambda_l1, 0) # it should be greater than 0
        params['lambda_l2'] = max(lambda_l2, 0) # it should be greater than 0
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        params['min_data_in_leaf'] = int(round(min_data_in_leaf))
        params['min_sum_heassian_in_leaf'] = int(round(min_sum_heassian_in_leaf))
        cv_result = lgb.cv(params, train_data, nfold=5, seed=random_seed, stratified=True, verbose_eval = verbose_eval, 
        metrics=['auc'])
        return max(cv_result['auc-mean'])
        
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (2,20),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50),
                                            'min_data_in_leaf':(10,100),
                                            'min_sum_heassian_in_leaf':(1,30)
                                            }, random_state=0)
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    # output optimization process
    if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")
    
    #garbage collect
    gc.collect()
    # return best parameters
    return lgbBO

#loading training and testing data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
features = [c for c in train.columns if c not in ['ID_code', 'target']]

#running bayesian hyperparameter optimization
opt_params = bayes_parameter_opt_lgbm(train[features], train['target'], init_round=10, opt_round=15, n_folds=5, random_seed=5, n_estimators=50000, learning_rate=0.0083, early_stopping_round = 1000, verbose_eval = 2500)

print(opt_params.res)

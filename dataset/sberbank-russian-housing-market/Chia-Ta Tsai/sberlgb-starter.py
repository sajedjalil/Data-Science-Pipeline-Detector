#utilities
from itertools import combinations
from math import exp, expm1, log1p, log10, log2, sqrt, ceil, floor, fmod
from random import choice, sample, shuffle, uniform
#pyData stack
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import skew, boxcox
#sklearn preprocessing, model selection
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation
from sklearn.metrics import mean_squared_error as scoring_mse, explained_variance_score as scoring_ev
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
#Gradient Boosting Machine
import lightgbm as lgb

#params
threads = -1
#target
target = 'price_doc'
target_id = 'id'
timestamp = 'timestamp'

###############################################################################
def rmsle(preds, train_data):
    labels = train_data.get_label()
    return 'rmsle', np.sqrt(np.mean((np.log1p(labels) - np.log1p(preds))**2)), False

def runLGB(train_X, train_y, test_X, test_y=None, train_w=None, seed_val=0, num_rounds=1000, min_rounds=25, opt_imba=False):
    
    opt_reg = True
    
    params = {}
    params['num_threads'] = 8
    params['boost'] = 'gbdt'
    #params['boost'] = 'dart'
    if opt_reg:
        #params['objective'] = 'regression_l1'
        #params['objective'] = 'regression_l2'
        params['objective'] = 'fair'
        #params['objective'] = 'poisson'
        #params['objective'] = 'huber'
        params['metric'] = 'rmse'
    else:
        params['objective'] = 'multiclass'
        params['is_unbalance'] = False
        params['num_class'] = len(np.unique(train_y))
        params['metric'] = 'multi_logloss'
    #learning
    params['learning_rate'] = 0.013
    params['num_leaves'] = 2 ** int(ceil(log1p(train_X.shape[1]) + 0))
    params['min_data_in_leaf'] = 10 ** 1 #default 100
    
    if params.get('boost') == 'dart':
        params['drop_rate'] = 0.1 #dart, deafault 0.1
        params['skip_drop'] = 0.5 #dart, deafault 0.5
        params['max_drop'] = 50 #dart, deafault 50
        params['uniform_drop'] = False #dart, deafault False
        params['xgboost_dart_mode'] = True #dart, deafault False
        
    #params['min_hessian'] = 10.0 #default 10.0
    params['feature_fraction'] = 0.75 #default=1.0
    params['bagging_fraction'] = 0.75 #default=1.0
    params['bagging_freq'] = 2
    params['lambda_l1'] = 0.002 #default 0
    params['lambda_l2'] = 0.005 #default 0
    params['data_random_seed'] = seed_val
    params['verbose'] = 0 #<0 = Fatel, =0 = Error(Warn), >0 = Info
    
    #metric
    params['metric_freq'] = 5 #deafult 1
    
    max_bin = 2 ** 13
    print('lightGBM w/ eta={} leaves={}'.format(params['learning_rate'], params['num_leaves']))

    if test_y is None:
        X_train = lgb.Dataset(train_X, label=train_y, weight=None, max_bin=max_bin, reference=None)
        gbm = lgb.train(params, X_train, valid_sets=[X_train], valid_names=['tr'],
                        num_boost_round=num_rounds, early_stopping_rounds=min_rounds)
#        y_pred = gbm.predict(test_X, num_iteration=gbm.best_iteration, pred_leaf=False)

    else:     
        X_train = lgb.Dataset(train_X, label=train_y, weight=None, max_bin=max_bin, reference=None, free_raw_data=False)
        X_valid = lgb.Dataset(test_X, label=test_y, reference=X_train, free_raw_data=False)
        gbm = lgb.train(params, X_train, valid_sets=[X_train, X_valid], valid_names=['tr', 'va'],
                        num_boost_round=num_rounds, early_stopping_rounds=min_rounds)

    return gbm


###############################################################################
if __name__ == '__main__':
    
    use_macro = False
    macro_df = pd.read_csv('../input/macro.csv')
    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')
    print('\ntrain={}, test={}'.format(train_df.shape, test_df.shape), flush=True, end='\n\n')

    #macro
    if use_macro:
        for f in macro_df.columns.tolist():
            if f in ['modern_education_share', 'child_on_acc_pre_school', 'old_education_build_share']:
                macro_df[f], uniques = pd.factorize(macro_df[f], sort=True)
            macro_df[f].fillna(method='backfill', inplace=True)
            macro_df[f].fillna(method='ffill', inplace=True)
    
    #quick clean
    train_size = len(train_df)
    train_test_df = pd.concat([train_df, test_df], ignore_index=True)
    for f in train_test_df.columns.tolist():
        if f not in [target, target_id, timestamp]:
            if train_test_df[f].dtype==np.object:
                train_test_df[f], uniques = pd.factorize(train_test_df[f], sort=True)
            else:
                #train_test_df[f].fillna(train_test_df[f].median(), inplace=True)
                train_test_df[f].fillna(method='backfill', inplace=True)

    del train_df, test_df
    if use_macro:
        train_test_df = train_test_df.merge(macro_df, how='left', on=timestamp)
    #date
    train_test_df[timestamp] = pd.to_datetime(train_test_df[timestamp])
    train_test_df['{}_m'.format(timestamp)] = train_test_df[timestamp].dt.month
    train_test_df['{}_md'.format(timestamp)] = train_test_df[timestamp].dt.day
    train_test_df['{}_wd'.format(timestamp)] = train_test_df[timestamp].dt.dayofweek
    train_test_df['{}_yd'.format(timestamp)] = train_test_df[timestamp].dt.dayofyear

    train_df, test_df = train_test_df[:train_size], train_test_df[train_size:]

    cut = 24284
    f_to_use = list(set(train_test_df.columns.tolist()).difference([target, target_id, timestamp]))
    X_train = train_df[f_to_use][:cut]
    y_train = train_df[target][:cut]
    X_valid = train_df[f_to_use][cut:]
    y_valid = train_df[target][cut:]
    print(X_train.shape, X_valid.shape)
    clf = runLGB(X_train, y_train.apply(log1p), X_valid, test_y=y_valid.apply(log1p), num_rounds=2000, min_rounds=200)
    
    test_X = test_df[f_to_use]
    y_pred = clf.predict(test_X, num_iteration=clf.best_iteration, pred_leaf=False)
    sub_test = pd.DataFrame({target: y_pred, target_id: test_df[target_id]})
    sub_test[target] = sub_test[target].apply(expm1)
    sub_test.to_csv('{}'.format('submission_SberLGB.csv'), index=False)
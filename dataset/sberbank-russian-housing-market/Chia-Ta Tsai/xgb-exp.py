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
import xgboost as xgb

#params
threads = -1
#target
target = 'price_doc'
target_id = 'id'
timestamp = 'timestamp'

###############################################################################
def rmsle(preds, train_data):
    labels = train_data.get_label()
    return 'rmsle', np.sqrt(np.mean((np.log1p(labels) - np.log1p(preds))**2))#, False

def runXGB(train_X, train_y, test_X, test_y=None, train_w=None, seed_val=2017, num_rounds=1000, min_rounds=25):
    
    params = {}
    params['silent'] = 1
    params['nthread'] = 8
    params['objective'] = 'reg:linear'
    #params['eval_metric'] = "rmse"
#    params['num_class'] = len(np.unique(train_y))#3
#    params['num_parallel_tree'] = len(np.unique(train_y))#3
    params['booster'] = 'gbtree'
#    params['booster'] = 'gblinear'
#    params['booster'] = 'dart'
#    params['tree_method'] = 'hist'
    params['max_bin'] = 2 ** 12
#    params['grow_policy'] = 'lossguide'
#    params['grow_policy'] = 'depthwise'
#    params['max_leaves'] = 2 ** int(ceil(log1p(train_X.shape[1]) + 1))
    params['rate_drop'] = 0.1
    params['skip_drop'] = 0.5
#    params['max_depth'] = 12
#    params['max_depth'] = 0
    params['max_depth'] = int(ceil(log1p(train_X.shape[1]) + 1))
    params['eta'] = 0.017
#    params['eta'] = 0.03
    params['subsample'] = 0.75
    params['colsample_bytree'] = 0.75
    params['colsample_bylevel'] = 1.0
    params['gamma'] = 0.05 #default=0
    params['lambda'] = 1.00 #default=1, l2 on weights
    params['alpha'] = 0.005 #default=0, l1 on weights
#    params['max_delta_step'] = 1 #1-10
#    params['min_child_weight'] = 64
    params['seed'] = seed_val
#    params['base_score'] = 16
    
    train_w = None
    
    print('xgb w/ eta={} depth={}'.format(params['eta'], params['max_depth']))
    #print('xgb w/ eta={} depth={}'.format(params['eta'], params['max_leaves']))

    xgtrain = xgb.DMatrix(train_X.as_matrix(), label=train_y, weight=train_w)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X.as_matrix(), label=test_y)
        watchlist = [(xgtrain, 'tr'), (xgtest, 'va')]
        gbm = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=min_rounds, feval=rmsle)
        #pred_test_y = model.predict(xgtest, ntree_limit=model.best_ntree_limit)
        del xgtrain, xgtest

    else:
        #xgtest = xgb.DMatrix(test_X.as_matrix())
        watchlist = [(xgtrain, 'tr')]
        gbm = xgb.train(params, xgtrain, num_rounds, watchlist)
        del xgtrain

    return gbm


###############################################################################
if __name__ == '__main__':
    
    use_macro = True
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
    #clf = runXGB(X_train, y_train.apply(log1p), X_valid, test_y=y_valid.apply(log1p), num_rounds=2000, min_rounds=100)
    clf = runXGB(X_train, y_train, X_valid, test_y=y_valid, num_rounds=2000, min_rounds=100)
    
    test_X = test_df[f_to_use]
    y_pred = clf.predict(xgb.DMatrix(test_X.as_matrix()), ntree_limit=clf.best_ntree_limit)
    #y_pred = clf.predict(test_X, num_iteration=clf.best_iteration, pred_leaf=False)
    sub_test = pd.DataFrame({target: y_pred, target_id: test_df[target_id]})
    #sub_test[target] = sub_test[target].apply(expm1)
    sub_test[target] = sub_test[target]
    sub_test.to_csv('{}'.format('submission_SberXGB.csv'), index=False)
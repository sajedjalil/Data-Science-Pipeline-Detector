#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 2018

@author: cttsai
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.model_selection import cross_val_score
import multiprocessing

def load_data():
    train_df = pd.read_csv("../input/train.csv").set_index("ID")
    print("Train rows and columns : {}".format(train_df.shape))

    test_df = pd.read_csv("../input/test.csv").set_index("ID")
    print("Test rows and columns : {}".format(test_df.shape))
    
    x_cols = test_df.columns.tolist()
    train_target = train_df['target'].apply(np.log1p)
    
    return train_df[x_cols], train_target, test_df[x_cols]

# data cleaning
# inspired from https://www.kaggle.com/samratp/aggregates-sumvalues-sumzeros-k-means-pca
def find_one_unique_columns(x):
    return [col for col in x.columns if len(x[col].unique()) < 2]

def find_sparse_columns(x, pct_threshold=0.98):
    return [col for col in x.columns if x[col].value_counts(normalize=True).loc[0] >= pct_threshold]

# aggregate
def func1(x):
    return sum(x == 0)
def func2(x):
    return np.sum(x)
def func3(x):
    return x.describe()
# inspired from https://www.kaggle.com/samratp/aggregates-sumvalues-sumzeros-k-means-pca
def aggregate(x, opt_pool=True):    
    if opt_pool:
        data = [f[1] for f in x.iterrows()]
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        sum_zero = pd.Series(pool.map(func1, data), index=x.index).rename('sum_zero')
        sum_non_zero = pd.Series(pool.map(func2, data), index=x.index).rename('sum_non_zero')
        desc = pd.concat(pool.map(func3, data), axis=1).T
        print(sum_zero.shape, sum_non_zero.shape, desc.shape) 
        pool.close()
        pool.join()
        del data
        return pd.concat([sum_zero, sum_non_zero, desc], axis=1)
    else:
        sum_zero = x.apply(lambda x: sum(x == 0), axis=1).rename('sum_zero')
        sum_non_zero = x.apply(lambda x: np.sum(x), axis=1).rename('sum_non_zero')
        desc = x.apply(lambda x: x.describe(), axis=1)
        return pd.concat([sum_zero, sum_non_zero, desc], axis=1)


def transform(X, tf, col_name):
    inds = X.index
    X = tf.fit_transform(np.nan_to_num(X.values)) 
    return pd.DataFrame(X, columns=['{}_{:03d}'.format(col_name, i+1) for i in range(X.shape[1])], index=inds)
    

def do_model_predict(x, y, test_x, params=None, nr_splits=5, random_state=42):
    
    preds_test = []
    preds_oof  = []
    kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
    for train_index, valid_index in kf.split(x):
        train_x, train_y  = x.iloc[train_index], y.iloc[train_index]
        valid_x, valid_y  = x.iloc[valid_index], y.iloc[valid_index]

        # XGB
        #params['base_score'] = y.mean()
        gbm = XGBRegressor(**params)
        gbm.fit(train_x, train_y,
                sample_weight=None, 
                eval_set=[(valid_x, valid_y)], 
                eval_metric='rmse', 
                early_stopping_rounds=25, 
                verbose=10, 
                xgb_model=None)

        preds_oof.append(pd.Series(np.expm1(gbm.predict(valid_x)), index=valid_x.index))
        preds_test.append(pd.Series(np.expm1(gbm.predict(test_x)), index=test_x.index))    

    preds_oof  = pd.concat(preds_oof, axis=0)    
    preds_test = pd.concat(preds_test, axis=1).mean(axis=1).squeeze() # keep in series

    print(preds_oof.shape, preds_test.shape)
    print(preds_oof.head())
    print(preds_test.head())
    return preds_oof, preds_test


def main():
    
    params = {
#        'max_depth': 12, 
#        'learning_rate': 0.02, 
#        'n_estimators': 1000, 
        'silent': True, 
        'objective': 'reg:linear', 
        'booster': 'gbtree', 
        'n_jobs': -1, 
#        'gamma': 0.002,
#        'min_child_weight': 5, 
#        'subsample': 0.8, 
#        'colsample_bytree': 0.25, 
#        'colsample_bylevel': 0.8, 
#        'reg_alpha': 0.002, 
#        'reg_lambda': 1.0, 
        'random_state': 42}
    
    #params['tree_method'] = 'hist'
    #params['grow_policy'] = 'lossguide'
#    params['max_leaves']  =  2 ** 6
    #params['base_score'] = 0.5
    
    # Prepare data
    X, y, test_X = load_data()
    cols_to_drop = find_one_unique_columns(X)
    sparse_cols = [f for f in find_sparse_columns(X) if f not in cols_to_drop]
    print(len(cols_to_drop))
    X      = X.drop(cols_to_drop, axis=1)
    test_X = test_X.drop(cols_to_drop, axis=1)
    print(X.shape, test_X.shape)
    
    # append X, test_X with transformed data
    print('transform X, test_X')
    tfs = {'pca': PCA(n_components=25, random_state=42),
           'ica': FastICA(n_components=25, random_state=42),
           'svd': TruncatedSVD(n_components=25, random_state=42),
           'grp': GaussianRandomProjection(n_components=25, eps=0.1, random_state=42),
           'srp': SparseRandomProjection(n_components=25, dense_output=True, random_state=42),
            }        
    if True:
        Xs = X.append(test_X)
        trans_Xs = [transform(Xs, v, col_name=k) for k, v in tfs.items()]
        trans_Xs.append(aggregate(Xs))
        
        X        = X.drop(sparse_cols, axis=1)
        test_X   = test_X.drop(sparse_cols, axis=1)
        print(X.shape, test_X.shape)
        # attach transformed features
        for trans_X in trans_Xs:
            X = X.join(trans_X, how='left')
            test_X = test_X.join(trans_X, how='left')
            print(X.shape, test_X.shape)
            del trans_X
    else:
        print('skip transform')
    
    # set base score for faster converge to set
    params['base_score'] = y.mean()
    
    # Find optimal parameters by cv
    from skopt.space import Real, Integer
    #from skopt.utils import use_named_args
    search_params_dict = {
        'max_depth': Integer(3, 10),
        'learning_rate': Real(10**-2, 10**-1, "log-uniform"),
        #'max_leaves': Integer(63,  191),
        'n_estimators': Integer(100, 500),
        'min_child_weight': Integer(2, 16),
        'gamma': Real(0.001,  0.5),
        'subsample': Real(0.7, 0.98),            
        'colsample_bytree': Real(0.5, 0.7),
        'colsample_bylevel': Real(0.5, 0.9),
        'reg_alpha': Real(10**-3, 10**-1, "log-uniform"),
        'reg_lambda': Real(10**-1, 10**1,  "log-uniform"),
    }
    
    print('HPO search range: ')
    search_params = []
    optimize_search_space = []
    for k, v in search_params_dict.items():
        optimize_search_space.append(v)
        search_params.append(k)
        print('{}: {}'.format(k, v))
    
    # evaluate function
    #@use_named_args(optimize_search_space)
    def objective(candidate_params):
        candidate_params = dict(zip(search_params, candidate_params)) 
        print(candidate_params)
        tuning_params = params.copy()
        tuning_params.update(candidate_params)
        model = XGBRegressor(**tuning_params)
        score = -np.mean(cross_val_score(model, X, y, cv=3, n_jobs=1, 
                                         scoring="neg_mean_squared_error"))
        del model
        return score                                                                                                                

    # do HPO
    from skopt import gp_minimize
    gp_optimizer = gp_minimize(objective, optimize_search_space, n_calls=25, random_state=42, verbose=True)

    optimized_params = {k: v for k, v in zip(search_params, gp_optimizer.x)}
    print('best cv score: {}'.format(gp_optimizer.fun))
    params.update(optimized_params)
    print('best params: {}'.format(params))
    
    # modeling and stacking
    preds_oofs, preds_test = [], []
    for s in [730, 920, 1020]:
    #for s in [1080]:
        oofs, test = do_model_predict(X, y, test_X, params=params, nr_splits=3, random_state=s)
        preds_oofs.append(oofs.rename('{}'.format(s)))
        preds_test.append(test.rename('{}'.format(s)))
        print('w/ params: {}'.format(params))
        
    #Out-of-fold Prediction
    mean_oof = pd.concat(preds_oofs, axis=1).mean(axis=1).squeeze().to_frame("target")
    mean_oof.index.name = "ID"
    mean_oof.to_csv("OOFpred_xgb_skopt.csv", index=True)
    #Out-of-Sample Prediction    
    mean_test = pd.concat(preds_test, axis=1).mean(axis=1).squeeze().to_frame("target")
    mean_test.index.name = "ID"
    mean_test.to_csv("subm_xgb_skopt.csv", index=True)
    

if __name__ == '__main__':
    main()

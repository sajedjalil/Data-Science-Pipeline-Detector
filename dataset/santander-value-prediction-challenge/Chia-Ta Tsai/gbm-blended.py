#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 2018

@author: cttsai
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from catboost import CatBoostRegressor


def load_data():
    train_df = pd.read_csv("../input/train.csv").set_index("ID")
    print("Train rows and columns : {}".format(train_df.shape))

    test_df = pd.read_csv("../input/test.csv").set_index("ID")
    print("Test rows and columns : {}".format(test_df.shape))
    
    x_cols = test_df.columns.tolist()
    train_target = train_df['target'].apply(np.log1p)
    
    return train_df[x_cols], train_target, test_df[x_cols]


def sparse_columns(x, pct_threshold=0.98):
    return [col for col in x.columns if x[col].value_counts(normalize=True).loc[0] >= pct_threshold]


def transform(X, tf, col_name):
    inds = X.index
    X = tf.fit_transform(X) 
    return pd.DataFrame(X, columns=['{}_{:03d}'.format(col_name, i+1) for i in range(X.shape[1])], index=inds)
    

def do_model_predict(x, y, test_x, params=None, nr_splits=5, random_state=42, model='xgb'):
    preds_test = []
    preds_oof  = []
    kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
    for train_index, valid_index in kf.split(x):
        train_x, train_y  = x.iloc[train_index], y.iloc[train_index]
        valid_x, valid_y  = x.iloc[valid_index], y.iloc[valid_index]

        # XGB
        if model == 'xgb':
            params['base_score'] = y.mean()
            params['random_state'] = random_state
            gbm = XGBRegressor(**params)
            gbm.fit(train_x, train_y,
                    sample_weight=None, 
                    eval_set=[(valid_x, valid_y)], 
                    eval_metric='rmse', 
                    early_stopping_rounds=25, 
                    verbose=10, 
                    xgb_model=None)
        #LGB
        elif model == 'lgb':
            params['base_score'] = y.mean()
            params['random_state'] = random_state
            gbm = LGBMRegressor(**params)
            gbm.fit(train_x, train_y,
                    sample_weight=None, 
                    #init_score=[y.mean()] * len(train_x),
                    eval_set=[(valid_x, valid_y)], 
                    eval_sample_weight=None,
                    #eval_init_score=[[y.mean()] * len(valid_x)],
                    eval_metric='rmse', 
                    early_stopping_rounds=25, 
                    verbose=10)

        elif model == 'cb':
            params['random_seed'] = random_state
            gbm = CatBoostRegressor(**params)
            gbm.fit(train_x, train_y, 
                    eval_set=(valid_x, valid_y), 
                    cat_features=[], 
                    use_best_model=True, 
                    verbose=True)

        preds_oof.append(pd.Series(gbm.predict(valid_x), index=valid_x.index).apply(np.expm1))
        preds_test.append(pd.Series(gbm.predict(test_x), index=test_x.index).apply(np.expm1))

    preds_test = pd.concat(preds_test, axis=1).mean(axis=1).squeeze() # keep in series
    preds_oof  = pd.concat(preds_oof, axis=0)
    print(preds_oof.shape, preds_test.shape)
    return preds_oof, preds_test


def main():
    
    xgb_params = {
        'max_depth': 13, 
        'learning_rate': 0.02, 
        'n_estimators': 1000, 
        'silent': True, 
        'objective': 'reg:linear', 
        'booster': 'gbtree', 
        'n_jobs': -1, 
        'gamma': 0.002,
        'min_child_weight': 5, 
        'subsample': 0.8, 
        'colsample_bytree': 0.7, 
        'colsample_bylevel': 0.7, 
        'reg_alpha': 0.002, 
        'reg_lambda': 1.0, 
        'random_state': 42}
    
    xgb_params['tree_method'] = 'hist'
    xgb_params['grow_policy'] = 'lossguide'
    xgb_params['max_leaves']  = 31
    #params['base_score'] = 0.5
    
    lgb_params = {
        'learning_rate': 0.03, 
        'n_estimators': 1000,
        'max_depth': 13, 
        'boosting_type': 'gbdt', 
        'objective': 'regression', 
        'metric': 'rmse', 
        'num_leaves': 31, 
        'min_split_gain': 0., 
        'min_child_weight': 0.003, 
        'min_child_samples': 20,
        'subsample': 0.9, 
        'subsample_freq': 5, 
        'colsample_bytree': 0.7,
        'reg_alpha': 0.001, 
        'reg_lambda': 0.003, 
        'random_state': 42}
    
    cb_params = {
        'iterations': 1000, 
        'learning_rate': 0.1, 
        'depth': 4, 
        'l2_leaf_reg': 20, 
        'bootstrap_type': 'Bernoulli', 
        'subsample': 0.6, 
        'eval_metric': 'RMSE', 
        'metric_period': 50, 
        'od_type':'Iter', 
        'od_wait': 45, 
        'random_seed': 42, 
        'allow_writing_files':False}
    
    
    X, y, test_X = load_data()
    cols_to_drop = sparse_columns(X)
    print(len(cols_to_drop))
    
    print('transform X, test_X')
    tfs = {#'pca': PCA(n_components=25, random_state=42),
           'svd': TruncatedSVD(n_components=25, random_state=42),
           #'ica': FastICA(n_components=25, random_state=42),
           #'grp': GaussianRandomProjection(n_components=25, eps=0.1, random_state=42),
           'srp': SparseRandomProjection(n_components=25, dense_output=True, random_state=42),
            }

    # append X, test_X with transformed data    
    trans_Xs = [transform(pd.concat([X, test_X]), v, col_name=k) for k, v in tfs.items()]    
    X        = X.drop(cols_to_drop, axis=1)
    test_X   = test_X.drop(cols_to_drop, axis=1)
    print(X.shape, test_X.shape)
    for trans_X in trans_Xs:
        X = X.join(trans_X, how='left')
        test_X = test_X.join(trans_X, how='left')
        print(X.shape, test_X.shape)
    
    # modeling and stacking
    X_stack, X_stack_test = [], []
    for m, params in {'xgb': xgb_params , 'lgb': lgb_params, 'cb': cb_params}.items():
        preds_oofs, preds_test = [], []
        for s in [2018, 1080]:
            oofs, test = do_model_predict(X, y, test_X, params=params, nr_splits=3, random_state=s, model=m)
            preds_oofs.append(oofs.rename('{}'.format(s)))
            preds_test.append(test.rename('{}'.format(s)))
        
        X_stack.append(pd.concat(preds_oofs, axis=1).mean(axis=1).squeeze().to_frame(m))
        X_stack_test.append(pd.concat(preds_test, axis=1).mean(axis=1).squeeze().to_frame(m))
        X_stack_test[-1].to_csv("subm_model_{}.csv".format(m), index=True)

    X_stack = pd.concat(X_stack, axis=1)
    y_stack = y.loc[X_stack.index]
    X_stack_test = pd.concat(X_stack_test, axis=1)
    print(X_stack.shape, X_stack_test.shape)
    
    # staked model
    model = SGDRegressor(loss='huber', 
                         penalty='elasticnet', 
                         alpha=0.0001, 
                         l1_ratio=0.15, 
                         max_iter=100, shuffle=True, 
                         random_state=42,)
    
    pl = PolynomialFeatures(degree=2, interaction_only=True)
    print(X_stack.head())
    print(y_stack.head())

    model.fit(np.log1p(pl.fit_transform(X_stack)), y_stack)
    subm = pd.DataFrame({'target': np.expm1(model.predict(np.log1p(pl.transform(X_stack_test)))), 
                         'ID': X_stack_test.index}).set_index('ID')
    subm.to_csv("subm_blend.csv", index=True)


if __name__ == '__main__':
    main()


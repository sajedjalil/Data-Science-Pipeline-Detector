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
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.random_projection import SparseRandomProjection
from sklearn.ensemble import RandomTreesEmbedding

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


def transform(x, test_x, tf, col_name):
    X = pd.concat([x, test_x])
    inds = X.index
    X = tf.fit_transform(X) 
    return pd.DataFrame(X, columns=['{}_{:03d}'.format(col_name, i+1) for i in range(X.shape[1])], index=inds)
    

def do_model_predict(x, y, test_x, params=None, nr_splits=5):
    
    params['base_score'] = y.mean()
    
    preds_test = []
    kf = KFold(n_splits=nr_splits, shuffle=True, random_state=42)
    for train_index, valid_index in kf.split(x):
        train_x, train_y  = x.iloc[train_index], y.iloc[train_index]
        valid_x, valid_y  = x.iloc[valid_index], y.iloc[valid_index]

        gbm = XGBRegressor(**params)
        gbm.fit(train_x, train_y,
                sample_weight=None, 
                eval_set=[(valid_x, valid_y)], 
                eval_metric='rmse', 
                early_stopping_rounds=10, 
                verbose=10, 
                xgb_model=None)

        preds_test.append(pd.Series(gbm.predict(test_x), index=test_x.index).apply(np.expm1))
    
    preds_test = pd.concat(preds_test, axis=1).mean(axis=1).squeeze().to_frame('target')
    print(preds_test.shape)
    return preds_test


def main():
    
    params = {
        'max_depth': 8, 
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
    
    params['tree_method'] = 'hist'
    params['grow_policy'] = 'lossguide'
    params['max_leaves']  = 31
    #params['base_score'] = 0.5
    
    X, y, test_X = load_data()
    cols_to_drop = sparse_columns(X)
    print(len(cols_to_drop))
    
    print('transform X, test_X')
    trans_Xs = []
    # svd
    tf = TruncatedSVD(n_components=10, random_state=42)
    trans_Xs.append(transform(X[X.columns], test_X[X.columns], tf, col_name='svd'))

    # rf
    rfe = RandomTreesEmbedding(n_estimators=100, max_depth=5, min_samples_split=20, sparse_output=False, n_jobs=-1, random_state=42)
    trans_Xs.append(transform(X[X.columns], test_X[X.columns], tf, col_name='rfe'))

    #grp
    tf = SparseRandomProjection(n_components=10, dense_output=True, random_state=42)
    trans_Xs.append(transform(X[cols_to_drop], test_X[cols_to_drop], tf, col_name='srp'))
    
    # append X, test_X with transformed data
    X = X.drop(cols_to_drop, axis=1)
    test_X = test_X.drop(cols_to_drop, axis=1)
    for trans_X in trans_Xs:
        X = X.join(trans_X, how='left')
        test_X = test_X.join(trans_X, how='left')
        print(X.shape, test_X.shape)
    
    test_preds = do_model_predict(X, y, test_X, params=params, nr_splits=5)
    print(test_preds.head())
    test_preds.to_csv("starter_xgb.csv", index=True)


if __name__ == '__main__':
    main()
    
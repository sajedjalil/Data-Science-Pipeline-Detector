"""
"""
import os
import gc
from functools import partial, wraps
from datetime import datetime as dt
import warnings

warnings.simplefilter('ignore', FutureWarning)

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import dask.dataframe as dd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

TARGET = 'target'
TARGET_INDEX = 'id'


def get_importances(clfs, feature_names):
    # Make importance dataframe
    ret = list()
    for i, model in enumerate(clfs, 1):
        # Feature importance
        imp_df = pd.DataFrame({
            "feature": feature_names,
            "gain": model.feature_importances_,
            # "fold": model.n_features_,
        })
        ret.append(imp_df)

    importance = pd.concat(ret, axis=0, sort=False)

    importance['gain_log'] = importance['gain']
    mean_gain = importance[['gain', 'feature']].groupby('feature').mean()
    importance['mean_gain'] = importance['feature'].map(mean_gain['gain'])
    # importance.to_csv('importance.csv', index=False)
    # plt.figure(figsize=(8, 12))
    # sns.barplot(x='gain_log', y='feature', data=importance.sort_values('mean_gain', ascending=False))
    return importance


def modeling_cross_validation(params, X, y, nr_folds=5):
    clfs = list()
    oof_preds = np.zeros(X.shape[0])
    # Split data with kfold
    kfolds = StratifiedKFold(n_splits=nr_folds, shuffle=False, random_state=42)
    for n_fold, (trn_idx, val_idx) in enumerate(kfolds.split(X, y)):
        X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

        print('{}, {}'.format(X_train.shape, y_train.shape))
        # LightGBM Regressor estimator
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[#(X_train, y_train,), 
                (X_valid, y_valid)],
            verbose=1000, eval_metric='auc',
            early_stopping_rounds=2500
        )

        clfs.append(model)
        oof_preds[val_idx] = model.predict_proba(X_valid, ntree_limit=model.best_ntree_limit)[:, -1]
        del X_train, X_valid; gc.collect()
        
    score = roc_auc_score(y, oof_preds)
    print(score)
    return clfs, score


def predict_cross_validation(test, clfs):
    sub_preds = np.zeros(test.shape[0])
    for i, model in enumerate(clfs, 1):
        test_preds = model.predict_proba(test, ntree_limit=model.best_ntree_limit)
        sub_preds += test_preds[:, 1]

    sub_preds = sub_preds / len(clfs)
    ret = pd.Series(sub_preds, index=test.index)
    ret.index.name = test.index.name
    return ret


def predict_test(test, features, clfs, dtypes, filename='tmp.csv'):
    preds_df = predict_cross_validation(test[features], clfs)
    preds_df = preds_df.to_frame(TARGET)
    preds_df.to_csv(filename, header=True, index=True)


def main():
    num_rows = None  # 20000
    dtypes = {}

    model_key = 'XGB'
    model_params = {
        'gpu_id': 0,
        #'n_gpus': 2,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'silent': True,
        'booster': 'gbtree',
        'n_jobs': -1,
        'n_estimators': 20000,
        'tree_method': 'gpu_hist',
        'grow_policy': 'lossguide',
        'max_depth': 8,
        'max_delta_step': 2,
        'seed': 538,
        'colsample_bylevel': 0.7,
        'colsample_bytree': 0.7,
        #'gamma': 0.1,
        'gamma': 1.5,
        'learning_rate': 0.05,
        'max_bin': 128,
        'max_leaves': 47,
        'min_child_weight': 16,
        #'reg_alpha': 0.05,
        'reg_alpha': 0.005,
        #'reg_lambda': 10.0,
        'reg_lambda': 1.0,
        #'subsample': 0.7,
        'subsample': 0.7,
    }

    train_features = list()
    train = pd.read_csv('../input/train.csv', nrows=num_rows, dtype=dtypes).set_index(TARGET_INDEX)
    test = pd.read_csv('../input/test.csv', nrows=num_rows, dtype=dtypes).set_index(TARGET_INDEX)
    
    enc = LabelEncoder().fit(train['wheezy-copper-turtle-magic'].tolist() + test['wheezy-copper-turtle-magic'].tolist())
    
    freq = train['wheezy-copper-turtle-magic'].value_counts()
    train['freq_wheezy-copper-turtle-magic'] = train['wheezy-copper-turtle-magic'].map(freq)
    test['freq_wheezy-copper-turtle-magic'] = test['wheezy-copper-turtle-magic'].map(freq)
    
    train['wheezy-copper-turtle-magic'] = enc.transform(train['wheezy-copper-turtle-magic'])
    test['wheezy-copper-turtle-magic'] = enc.transform(test['wheezy-copper-turtle-magic'])
    train_features = list(filter(lambda f: f != TARGET, train.columns.tolist()))

    model_params['base_score'] = train[TARGET].mean()

    # modeling
    train_x = train[train_features]
    print(train_x.shape)
    clfs, score = modeling_cross_validation(model_params, train_x, train[TARGET], nr_folds=5)

    #file_stem = '{:.6f}_{}_{}'.format(score, model_key, dt.now().strftime('%Y-%m-%d-%H-%M'))
    #filename = 'subm_{}.csv'.format(file_stem)
    filename = 'submission.csv'
    predict_test(test, train_features, clfs, dtypes, filename=filename)

    # train_features = list(filter(lambda f: f != TARGET, train_x.columns.tolist()))
    #imp = get_importances(clfs, feature_names=train_features)
    #imp.to_csv('importance_{}.csv'.format(file_stem), index=False)


if __name__ == '__main__':
    main()
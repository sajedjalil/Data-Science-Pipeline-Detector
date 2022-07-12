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

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

TARGET = 'target'
TARGET_INDEX = 'ID_code'


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

        # LightGBM Regressor estimator
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=200, eval_metric='auc',
            early_stopping_rounds=100
        )

        clfs.append(model)
        oof_preds[val_idx] = model.predict_proba(X_valid, ntree_limit=model.best_ntree_limit)[:, -1]

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


def predict_test_chunk(features, clfs, dtypes, filename='tmp.csv', chunks=100000):
    for i_c, df in enumerate(pd.read_csv('../input/test.csv',
                                         chunksize=chunks,
                                         dtype=dtypes,
                                         iterator=True)):

        df.set_index(TARGET_INDEX, inplace=True)
        preds_df = predict_cross_validation(df[features], clfs)
        preds_df = preds_df.to_frame(TARGET)

        if i_c == 0:
            preds_df.to_csv(filename, header=True, mode='a', index=True)
        else:
            preds_df.to_csv(filename, header=False, mode='a', index=True)

        del preds_df
        gc.collect()


def main():
    num_rows = None  # 20000
    dtypes = {}

    model_key = 'XGB'
    model_params = {
        'gpu_id': 0,
        # 'n_gpus': 3,
        'objective': 'binary:logitraw',
        'eval_metric': 'auc',
        'silent': True,
        'booster': 'gbtree',
        'n_jobs': 4,
        'n_estimators': 2500,
        'tree_method': 'gpu_hist',
        #'tree_method': 'hist',
        'grow_policy': 'lossguide',
        'max_delta_step': 2,
        'seed': 538,
        'colsample_bylevel': 0.9,
        'colsample_bytree': 0.8,
        'gamma': 100.0,
        'learning_rate': 0.1,
        'max_bin': 64,
        'max_depth': 8,
        'max_leaves': 15,
        'min_child_weight': 16,
        'reg_alpha': 1e-06,
        'reg_lambda': 10.0,
        'subsample': 0.7}

    train_features = list()
    train = pd.read_csv('../input/train.csv', nrows=num_rows, dtype=dtypes).set_index(TARGET_INDEX)
    train_features = list(filter(lambda f: f != TARGET, train.columns.tolist()))

    # modeling
    clfs, score = modeling_cross_validation(model_params, train[train_features], train[TARGET])

    file_stem = '{:.6f}_{}_{}'.format(score, model_key, dt.now().strftime('%Y-%m-%d-%H-%M'))
    filename = 'subm_{}.csv'.format(file_stem)
    predict_test_chunk(train_features, clfs, dtypes, filename=filename, chunks=100000)

    imp = get_importances(clfs, feature_names=train_features)
    imp.to_csv('importance_{}.csv'.format(file_stem), index=False)


if __name__ == '__main__':
    main()
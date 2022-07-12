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

from catboost import Pool, CatBoostClassifier

TARGET = 'target'
TARGET_INDEX = 'ID_code'


class Featurize(object):
    def __init__(self):
        self.bins = np.linspace(0., 1.0, num=101).tolist()
        self.binning = None

    @staticmethod
    def _calculate_series(df, prefix):

        orig_columns = df.columns.tolist()
        df = df.copy()

        df['min'] = df.min(axis=1)
        df['max'] = df.max(axis=1)

        df['mean'] = df.mean(axis=1)
        df['median'] = df.median(axis=1)
        df['std'] = df.std(axis=1)
        df['sem'] = df.sem(axis=1)

        df['skew'] = df.skew(axis=1)
        df['kurt'] = df.kurtosis(axis=1)

        df['spread'] = df['max'] - df['min']
        df['spread_over_median'] = df['spread'] / df['median']
        df['spread_over_std'] = df['spread'] / df['std']
        df['zscore_median'] = (df['median'] - df['mean']) / df['std']

        new_stats_cols = list(filter(lambda x: x not in orig_columns, df.columns.tolist()))
        df = df.reindex(columns=new_stats_cols)
        df.rename(columns={k: '_'.join([prefix, k]) for k in new_stats_cols}, inplace=True)
        return df

    def fit(self, df):
        self.binning = {f: pd.qcut(df[f], self.bins, retbins=True)[1] for f in df.columns}

    def transform(self, df):

        reductions = list()
        reductions.append(df)

        df_bins = df.apply(lambda x: pd.cut(x, bins=self.binning[x.name]).cat.codes)
        df_bins.rename(columns={k: 'bins_' + k for k in df_bins.columns}, inplace=True)
        reductions.append(df_bins)

        # bins = [float(s) for s in range(-10, 11, 5)]
        # reductions.append(df.apply(lambda x: x.value_counts(bins=bins), axis=1))

        df_stats = self._calculate_series(df, prefix='stats')
        reductions.append(df_stats)

        periods = [1, 2, 4, 8]
        for f in periods:
            df_diff = df.diff(periods=f, axis=1)
            df_diff = self._calculate_series(df_diff, prefix='diff_n{}'.format(f))
            reductions.append(df_diff)

        periods = [1, 2, 4, 8]
        for f in periods:
            df_diff = df.pct_change(periods=f, axis=1)
            df_diff = self._calculate_series(df_diff, prefix='pct_chg_n{}'.format(f))
            reductions.append(df_diff)

        periods = [0.5, 0.3, 0.1]
        for f in periods:
            df_ema = df.ewm(com=f, axis=1).mean()
            df_ema = self._calculate_series(df_ema, prefix='ema_com{:.02f}'.format(f))
            reductions.append(df_ema)

        df = pd.concat(reductions, axis=1)
        # pint(df.head().T)

        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


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


def predict_cross_validation(test, clfs, featurize):
    if featurize:
        test = featurize.transform(test)

    sub_preds = np.zeros(test.shape[0])
    for i, model in enumerate(clfs, 1):
        test_preds = model.predict_proba(test)
        sub_preds += test_preds[:, 1]

    sub_preds = sub_preds / len(clfs)
    ret = pd.Series(sub_preds, index=test.index)
    ret.index.name = test.index.name
    return ret


def predict_test_chunk(features, clfs, dtypes, featurize=None, filename='tmp.csv', chunks=100000):
    for i_c, df in enumerate(pd.read_csv('../input/test.csv',
                                         chunksize=chunks,
                                         dtype=dtypes,
                                         iterator=True)):

        df.set_index(TARGET_INDEX, inplace=True)

        preds_df = predict_cross_validation(df[features], clfs, featurize)
        preds_df = preds_df.to_frame(TARGET)

        if i_c == 0:
            preds_df.to_csv(filename, header=True, mode='a', index=True)
        else:
            preds_df.to_csv(filename, header=False, mode='a', index=True)

        del preds_df
        gc.collect()


def modeling_cross_validation(params, X, y, nr_folds=5):
    clfs = list()
    oof_preds = np.zeros(X.shape[0])
    # Split data with kfold
    kfolds = StratifiedKFold(n_splits=nr_folds, shuffle=False, random_state=42)
    for n_fold, (trn_idx, val_idx) in enumerate(kfolds.split(X, y)):
        X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

        model = CatBoostClassifier(loss_function="Logloss",
                                   eval_metric="AUC",
                                   task_type="GPU",
                                   learning_rate=0.005,
                                   iterations=200000,
                                   l2_leaf_reg=10,
                                   random_seed=42,
                                   od_type="Iter",
                                   depth=10,
                                   early_stopping_rounds=2000
                                  )
                # LightGBM Regressor estimator
        #model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=1000, #eval_metric='auc',
            early_stopping_rounds=2500
        )

        clfs.append(model)
        oof_preds[val_idx] = model.predict_proba(X_valid)[:, -1]

    score = roc_auc_score(y, oof_preds)
    print(score)
    return clfs, score

        
def main():
    num_rows = None  # 20000
    dtypes = {}

    model_key = 'CB'
    model_params = {}

    train_features = list()
    train = pd.read_csv('../input/train.csv', nrows=num_rows, dtype=dtypes).set_index(TARGET_INDEX)
    train_features = list(filter(lambda f: f != TARGET, train.columns.tolist()))

    # modeling
    featurize = Featurize()
    train_x = featurize.fit_transform(train[train_features])
    
    clfs, score = modeling_cross_validation(model_params, train_x, train[TARGET], nr_folds=11)

    file_stem = '{:.6f}_{}_{}'.format(score, model_key, dt.now().strftime('%Y-%m-%d-%H-%M'))
    filename = 'subm_{}.csv'.format(file_stem)
    predict_test_chunk(train_features, clfs, dtypes, featurize, filename=filename, chunks=100000)

    train_features = list(filter(lambda f: f != TARGET, train_x.columns.tolist()))
    imp = get_importances(clfs, feature_names=train_features)
    imp.to_csv('importance_{}.csv'.format(file_stem), index=False)


if __name__ == '__main__':
    main()

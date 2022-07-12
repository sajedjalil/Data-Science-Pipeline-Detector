"""
"""
import os
import gc
from multiprocessing import Pool
from functools import partial, wraps
from datetime import datetime as dt
import warnings

warnings.simplefilter('ignore', FutureWarning)

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import dask.dataframe as dd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

TARGET = 'target'
TARGET_INDEX = 'ID_code'

NUM_CPU = 16


def kde_fit(x):
    data, k, v = x
    return k, v.fit(data)


def kde_score(x):
    data, k, v = x
    return k, v.score_samples(data)


class Featurize(object):
    def __init__(self):
        self.bins = np.linspace(0., 1.0, num=201).tolist()

        self.binning = None

        self.freq = None
        self.freq_pos = None
        self.freq_neg = None

        self.cols_kde = [
            'var_12', 'var_75', 'var_81', 'var_83', 'var_89',
            'var_95', 'var_99', 'var_113', 'var_122', 'var_139',
            'var_154', 'var_190',]

        self.kde_estimators_pos = {
            f: KernelDensity(bandwidth=0.75, ) for f in self.cols_kde}
        self.kde_estimators_neg = {
            f: KernelDensity(bandwidth=0.75, ) for f in self.cols_kde}

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
        return df.astype(np.float32)

    def fit(self, df, y):
        ret = {f: pd.qcut(df[f], self.bins, retbins=True, duplicates='drop') for f in df.columns}

        #mask_pos = y == 1
        #ret = {f: pd.qcut(df.loc[mask_pos, f], self.bins, retbins=True, duplicates='drop') for f in df.columns}
        #print(df.loc[mask_pos].apply(lambda x: x.nunique()).describe())

        self.binning = {k: v[1] for k, v in list(ret.items())}
        self.freq = {k: v[0].value_counts(normalize=True) for k, v in list(ret.items())}

        if y is not None:

            mask_pos = y == 1
            mask_neg = y == 0

            self.freq_pos = {
                k: v[0].loc[mask_pos].value_counts(normalize=True) for k, v in list(ret.items())}

            self.freq_neg = {
                k: v[0].loc[mask_neg].value_counts(normalize=True) for k, v in list(ret.items())}

            p = Pool(NUM_CPU)

            sel_df = df.loc[mask_pos]
            print(sel_df.shape)

            queue = [(sel_df[[k]], k, v) for k, v in list(self.kde_estimators_pos.items())]
            ret = list(p.map(kde_fit, queue))
            self.kde_estimators_pos = {k: v for k, v in ret}

            sel_df = df.loc[mask_neg]
            print(sel_df.shape)

            queue = [(sel_df[[k]], k, v) for k, v in list(self.kde_estimators_neg.items())]
            ret = list(p.map(kde_fit, queue))
            self.kde_estimators_neg = {k: v for k, v in ret}

            p.close()
            p.join()
        print('fitted')
        
    def transform(self, df):

        reductions = list()
        reductions.append(df.astype(np.float32))

        df_bins = df.apply(lambda x: pd.cut(x, bins=self.binning[x.name]).cat.codes)
        df_bins.rename(columns={k: 'bins_' + k for k in df_bins.columns}, inplace=True)
        # reductions.append(df_bins)

        df_bin_stats = self._calculate_series(df_bins, prefix='bin_stats')
        reductions.append(df_bin_stats)

        #
        df_bins_freq = df_bins.copy()
        for k, v in list(self.freq.items()):
            df_bins_freq['bins_' + k] = df_bins_freq['bins_' + k].map(v)

        #
        df_bins_freq_pos = df_bins.copy()
        for k, v in list(self.freq_pos.items()):
            df_bins_freq_pos['bins_' + k] = df_bins_freq_pos['bins_' + k].map(v)

        #
        cols = df_bins_freq_pos.columns.intersection(df_bins_freq.columns)

        df_bins_diff = pd.concat([df_bins_freq_pos[f] - df_bins_freq[f] for f in cols], axis=1)
        df_bins_diff.rename(columns={k: 'bins_diff_' + k for k in df_bins_diff.columns}, inplace=True)
        # reductions.append(df_bins_diff)
        df_bin_stats = self._calculate_series(df_bins_diff, prefix='bin_diff_stats')
        reductions.append(df_bin_stats)

        df_bins_ratio = pd.concat([df_bins_freq_pos[f] / df_bins_freq[f] for f in cols], axis=1)
        df_bins_ratio.rename(columns={k: 'bins_ratio_' + k for k in df_bins_ratio.columns}, inplace=True)
        # reductions.append(df_bins_ratio)
        df_bin_stats = self._calculate_series(df_bins_ratio, prefix='bin_ratio_stats')
        reductions.append(df_bin_stats)
        #

        df_bin_stats = self._calculate_series(df_bins_freq, prefix='bin_freq_stats')
        reductions.append(df_bin_stats)

        df_bins_freq.rename(columns={k: 'freq_' + k for k in df_bins.columns}, inplace=True)
        # reductions.append(df_bins_freq)

        df_bin_stats_pos = self._calculate_series(df_bins_freq_pos, prefix='bin_freq_stats_pos')
        reductions.append(df_bin_stats_pos)

        df_bins_freq_pos.rename(columns={k: 'freq_pos_' + k for k in df_bins.columns}, inplace=True)
        reductions.append(df_bins_freq_pos)

        df = pd.concat(reductions, axis=1)
        print(df.shape)

        if True:
            p = Pool(NUM_CPU)

            queue = [(df[[k]], k, v) for k, v in list(self.kde_estimators_pos.items())]
            ret = list(p.map(kde_score, queue))
            for k, v in ret:
                df['kde_pos_' + k] = v

            p.close()
            p.join()

        return df.astype(np.float32)

    def fit_transform(self, df, y=None):
        self.fit(df, y)
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


def modeling_cross_validation(params, X, y, nr_folds=5):
    featurizes = list()
    clfs = list()
    oof_preds = np.zeros(X.shape[0])
    # Split data with kfold
    kfolds = StratifiedKFold(n_splits=nr_folds, shuffle=False, random_state=42)
    for n_fold, (trn_idx, val_idx) in enumerate(kfolds.split(X, y)):
        X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]
        print('{}, {}'.format(X_train.shape, y_train.shape))

        # modeling
        featurize = Featurize()
        X_train = featurize.fit_transform(X_train, y_train)
        X_valid = featurize.transform(X_valid)

        print('{}, {}'.format(X_train.shape, y_train.shape))
        X_train, y_train = upsampling(X_train, y_train, pos=4, neg=1)
        print('upsampled {}, {}'.format(X_train.shape, y_train.shape))
        
        # LightGBM Regressor estimator
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[
                (X_train, y_train,),
                (X_valid, y_valid)],
            verbose=1000, eval_metric='auc',
            early_stopping_rounds=2500
        )

        featurizes.append(featurize)
        clfs.append(model)
        oof_preds[val_idx] = model.predict_proba(
            X_valid, ntree_limit=model.best_ntree_limit)[:, -1]
        del X_train, X_valid; gc.collect()
        
    score = roc_auc_score(y, oof_preds)
    print(score)
    return featurizes, clfs, score


def predict_cross_validation(test, featurizes, clfs):
    sub_preds = np.zeros(test.shape[0])
    for i, (featurize, model) in enumerate(zip(featurizes, clfs), 1):
        test = featurize.transform(test)

        test_preds = model.predict_proba(test, ntree_limit=model.best_ntree_limit)
        sub_preds += test_preds[:, 1]

    sub_preds = sub_preds / len(clfs)
    ret = pd.Series(sub_preds, index=test.index)
    ret.index.name = test.index.name
    return ret


def predict_test_chunk(features, featurizes, clfs, dtypes, filename='tmp.csv', chunks=100000):
    for i_c, df in enumerate(pd.read_csv('../input/test.csv',
                                         chunksize=chunks,
                                         dtype=dtypes,
                                         iterator=True)):

        df.set_index(TARGET_INDEX, inplace=True)

        preds_df = predict_cross_validation(df[features], featurizes, clfs)
        preds_df = preds_df.to_frame(TARGET)

        if i_c == 0:
            preds_df.to_csv(filename, header=True, mode='a', index=True)
        else:
            preds_df.to_csv(filename, header=False, mode='a', index=True)

        del preds_df
        gc.collect()


def shuffle(df, rng):
    ddf = dd.from_pandas(df, chunksize=1000)
    ddf = ddf.apply(
        lambda x: pd.Series(x.sample(frac=1., random_state=rng).values), axis=1).compute()
    ddf.columns = df.columns
    return ddf   


def upsampling_one_class(li_df, df, df_label, freq, rng=None):
    for i in range(freq):
        if rng is None:
            tmp = df.copy()
        else:
            tmp = shuffle(df, rng)

        tmp[TARGET] = df_label
        li_df.append(tmp)

    return li_df


def upsampling(x, y, pos=4, neg=1, pos_shuffle=0, neg_shuffle=0):
    
    mask = y == 0
    x_neg = x.loc[mask]
    x_pos = x.loc[~mask]    

    augment_data = list()
    augment_data.append(pd.concat([x, y], axis=1))
    
    rng = np.random.RandomState(42)

    augment_data = upsampling_one_class(augment_data, x_neg, 0, freq=neg_shuffle, rng=rng)
    augment_data = upsampling_one_class(augment_data, x_pos, 1, freq=pos_shuffle, rng=rng)

    augment_data = upsampling_one_class(augment_data, x_neg, 0, freq=neg)
    augment_data = upsampling_one_class(augment_data, x_pos, 1, freq=pos)

    x_all = pd.concat(augment_data).sort_index()
    x = x_all.reindex(columns=x.columns)
    return x, x_all[TARGET] 


def main():
    num_rows = None  # 20000
    dtypes = {}

    model_key = 'XGB'
    model_params = {
        'gpu_id': 0,
        'n_gpus': 2,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'silent': True,
        'booster': 'gbtree',
        'n_jobs': -1,
        'n_estimators': 25000,
        'tree_method': 'gpu_hist',
        'grow_policy': 'lossguide',
        'max_depth': 9,
        'max_delta_step': 2,
        'seed': 538,
        'colsample_bylevel': 0.7,
        'colsample_bytree': 0.8,
        #'gamma': 0.1,
        'gamma': 0.5,
        'learning_rate': 0.02,
        'max_bin': 64,
        'max_leaves': 47,
        'min_child_weight': 64,
        #'reg_alpha': 0.05,
        'reg_alpha': 5.,
        #'reg_lambda': 10.0,
        'reg_lambda': 5.0,
        #'subsample': 0.7,
        'subsample': 0.5,
    }

    train_features = list()
    train = pd.read_csv('../input/train.csv', nrows=num_rows, dtype=dtypes).set_index(TARGET_INDEX)
    train_features = list(filter(lambda f: f != TARGET, train.columns.tolist()))

    # initialize bias
    model_params['base_score'] = train[TARGET].mean()

    featurizes, clfs, score = modeling_cross_validation(
        model_params, train[train_features], train[TARGET], nr_folds=5)

    file_stem = '{:.6f}_{}_{}'.format(score, model_key, dt.now().strftime('%Y-%m-%d-%H-%M'))
    filename = 'subm_{}.csv'.format(file_stem)
    predict_test_chunk(strain_features, featurizes, clfs, dtypes, filename=filename, chunks=10000)

    #train_features = list(filter(lambda f: f != TARGET, train_x.columns.tolist()))
    #imp = get_importances(clfs, feature_names=train_features)
    #imp.to_csv('importance_{}.csv'.format(file_stem), index=False)


if __name__ == '__main__':
    main()

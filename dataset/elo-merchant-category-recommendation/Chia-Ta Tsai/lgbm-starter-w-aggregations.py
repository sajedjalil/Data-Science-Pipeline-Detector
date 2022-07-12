""""
referenced from the following excelent kernels
https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-elo
https://www.kaggle.com/truocpham/feature-engineering-and-lightgbm-starter
https://www.kaggle.com/rooshroosh/simple-data-exploration-with-python-lb-3-760
https://www.kaggle.com/youhanlee/hello-elo-ensemble-will-help-you
https://www.kaggle.com/ashishpatel26/beginner-guide-of-elo-eda-kfold-lightgbm
https://www.kaggle.com/kailex/tidy-elo-starter-3-813
https://www.kaggle.com/peterhurford/you-re-going-to-want-more-categories-lb-3-737
"""
import os
import gc
from functools import partial, wraps
from datetime import datetime as dt
import warnings
warnings.simplefilter('ignore', FutureWarning)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, TimeSeriesSplit

import lightgbm as lgb


def missing_impute(df, impute_dict=None):
    if impute_dict is None:
        df_mean = df.replace(to_replace=[np.inf, -np.inf], value=np.nan).mean()
        impute_dict = dict()
        impute_dict.update({k: 'other' for k in df.select_dtypes(include="object")})
        impute_dict.update({k: df_mean[k] for k in df.select_dtypes(include=["int64", "float64"])})

        if impute_dict:
            print('impute:\n{}'.format(pd.Series(impute_dict)))

    elif not isinstance(impute_dict, dict):
        raise TypeError

    df.fillna(impute_dict, inplace=True)
    return df, impute_dict


def datetime_extract(df, dt_col='first_active_month', reference_date=None):
    df['date'] = df[dt_col].dt.date
    df['day'] = df[dt_col].dt.day
    df['dayofweek'] = df[dt_col].dt.dayofweek
    df['dayofyear'] = df[dt_col].dt.dayofyear
    df['days_in_month'] = df[dt_col].dt.days_in_month
    df['daysinmonth'] = df[dt_col].dt.daysinmonth
    df['month'] = df[dt_col].dt.month
    df['week'] = df[dt_col].dt.week
    df['weekday'] = df[dt_col].dt.weekday
    df['weekofyear'] = df[dt_col].dt.weekofyear
    # df['year'] = df[dt_col].dt.year
    if reference_date:
        df['elapsed_time'] = (reference_date - df[dt_col]).dt.days

    return df


def aggregate_interaction(df, groupby_key, interact_keys, aggregations, stem):
    # purchase_date
    if 'purchase_date' in df.columns:
        df.loc[:, 'purchase_date'] = pd.DatetimeIndex(df['purchase_date']).astype(np.int64) * 1e-9
    
    grouped = df.groupby(interact_keys)
    intermediate_group = grouped.agg(aggregations)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)

    final_group = intermediate_group.groupby('card_id').agg(['median', 'std'])
    final_group.columns = pd.Index(['{}'.format('_'.join([stem] + list(e))) for e in final_group.columns.tolist()])
    final_group.reset_index(inplace=True)
    
    return final_group


def aggregate_df(df, groupby_key, aggregations, stem):
    """
    groupby_key
    """
    agg_df = df.groupby(groupby_key).agg(aggregations)
    agg_df.columns = pd.Index(['{}'.format('_'.join([stem] + list(e))) for e in agg_df.columns.tolist()])
    agg_df.index.name = groupby_key
    agg_df.reset_index(inplace=True)

    return agg_df


def process_transactions(df, df_key, configs, agg_categorical=True):
    # purchase_date
    if 'purchase_date' in df.columns:
        df.loc[:, 'purchase_date'] = pd.DatetimeIndex(df['purchase_date']).astype(np.int64) * 1e-9

    if agg_categorical:
        cat_cols = [f for f in df.columns if f.startswith('category_')]
        print('find {}: {}'.format(len(cat_cols), cat_cols))
        agg_cols = ['mean', 'var']
        old_cols = df.columns.tolist()
        df = pd.get_dummies(df, columns=cat_cols)
        new_cols = sorted(list(set(df.columns.tolist()).difference(old_cols)))
        configs.update({c: agg_cols for c in new_cols})
        print('adding categorical: {}'.format(new_cols))

    df_agg = aggregate_df(df, groupby_key='card_id', aggregations=configs, stem=df_key)
    print(df_agg.describe().T)

    fg_grouped = (df.groupby('card_id').size().reset_index(name='{}_transsact_count'.format(df_key)))

    ret = pd.merge(fg_grouped, df_agg, on='card_id', how='left')
    return ret


def process_count(df, features=None, dfs_to_join=None):
    if dfs_to_join is None:
        dfs_to_join = dict()
        if features is not None:
            dfs_to_join = {f: df[f].value_counts().rename('{}_counts'.format(f)) for f in features}

    for k, v in dfs_to_join.items():
        df = df.merge(v, on=k)

    return df.sort_index(1), dfs_to_join


def modeling_cross_validation(params, X, y, nr_folds=5):
    clfs = list()
    oof_preds = np.zeros(X.shape[0])
    # Split data with kfold
    kfolds = TimeSeriesSplit(n_splits=nr_folds)
    for n_fold, (trn_idx, val_idx) in enumerate(kfolds.split(X, y)):
        X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

        # LightGBM Regressor estimator
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=25, eval_metric='rmse',
            early_stopping_rounds=100
        )

        clfs.append(model)
        oof_preds[val_idx] = model.predict(X_valid, num_iteration=model.best_iteration_)

    score = mean_squared_error(y, oof_preds) ** .5
    return clfs, score


def get_importances(clfs, feature_names):
    # Make importance dataframe
    importances = pd.DataFrame()
    for i, model in enumerate(clfs, 1):
        # Feature importance
        imp_df = pd.DataFrame({
                "feature": feature_names, 
                "gain": model.booster_.feature_importance(importance_type='gain'),
                "fold": model.n_features_,
                })
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

    importances['gain_log'] = importances['gain']
    mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
    importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])
    importances.to_csv('importance.csv', index=False)
    # plt.figure(figsize=(8, 12))
    # sns.barplot(x='gain_log', y='feature', data=importances.sort_values('mean_gain', ascending=False))
    return importances


def predict_cross_validation(test, clfs):
    sub_preds = np.zeros(test.shape[0])
    for i, model in enumerate(clfs, 1):    
        test_preds = model.predict(test, num_iteration=model.best_iteration_)
        sub_preds += test_preds

    sub_preds = sub_preds / len(clfs)

    sample_submission = pd.read_csv('../input/sample_submission.csv')
    sample_submission['target'] = sub_preds
    return sample_submission


def main():

    aggregations = {
        'historical': {
            'authorized_flag': ['sum', 'mean'],
            'merchant_id': ['nunique'],
            'city_id': ['nunique'],
            'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
            'installments': ['sum', 'median', 'max', 'min', 'std'],
            'purchase_date': [np.ptp],
            'month_lag': ['min', 'max']
        },

        'new_merchant': {
            'authorized_flag': ['sum', 'mean'],
            'merchant_id': ['nunique'],
            'city_id': ['nunique'],
            'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
            'installments': ['sum', 'median', 'max', 'min', 'std'],
            'month_lag': ['min', 'max']
        },
    }

    model_params = {
        'device': 'cpu', 
        'objective': 'regression', 
        'boosting_type': 'dart', 
        'n_jobs': -1, 
        'max_depth': 7, 
        'n_estimators': 2500, 
        'subsample_freq': 2, 
        'subsample_for_bin': 200000, 
        'min_data_per_group': 100, 
        'max_cat_to_onehot': 4, 
        'cat_l2': 86.80, 
        'cat_smooth': 0.335, 
        'max_cat_threshold': 32, 
        'metric_freq': 10, 
        'verbosity': -1, 
        'metric': 'rmse', 
        'xgboost_dart_mode': False, 
        'uniform_drop': False, 
        'colsample_bytree': 0.506, 
        'drop_rate': 0.34, 
        'learning_rate': 0.064, 
        'max_drop': 12, 
        'min_child_samples': 98, 
        'min_child_weight': 3.973, 
        'min_split_gain': 1.9424665337111478e-05, 
        'num_leaves': 22, 
        'reg_alpha': 1.6097, 
        'reg_lambda': 0.00110, 
        'skip_drop': 0.2576, 
        'subsample': 0.88}

        
    #
    df_key = 'historical'
    historical_transactions = pd.read_csv('../input/historical_transactions.csv')
    historical_transactions['authorized_flag'] = historical_transactions['authorized_flag'].map({'Y': 1, 'N': 0})
    df_agg_historical_interact = aggregate_interaction(historical_transactions, 
                                                    groupby_key='card_id', 
                                                    interact_keys=['card_id', 'month_lag'], 
                                                    aggregations=aggregations[df_key], 
                                                    stem=df_key)
    df_agg_historical = process_transactions(
        historical_transactions, df_key, configs=aggregations[df_key],)
    print(df_agg_historical.shape)

    #
    df_key = 'new_merchant'
    new_merchant_transactions = pd.read_csv('../input/new_merchant_transactions.csv')
    # a bit cleaning
    new_merchant_transactions['authorized_flag'] = new_merchant_transactions['authorized_flag'].map({'Y': 1, 'N': 0})
    df_agg_new_merchant = process_transactions(
        new_merchant_transactions, df_key, configs=aggregations[df_key],)
    print(df_agg_new_merchant.shape)

    # train
    train = pd.read_csv('../input/train.csv', parse_dates=["first_active_month"])
    train, dict_impute = missing_impute(train)
    train = datetime_extract(train, dt_col='first_active_month', reference_date=train['first_active_month'].max())

    train = pd.merge(train, df_agg_historical, on="card_id", how="left")
    train = pd.merge(train, df_agg_new_merchant, on="card_id", how="left")
    train = pd.merge(train, df_agg_historical_interact, on="card_id", how="left")

    excluded_features = ['first_active_month', 'card_id', 'target', 'date']
    train_features = [c for c in train.columns if c not in excluded_features]
    categorical_features = [f for f in train.select_dtypes(include="object") if f in train_features]
    train, counts_dfs = process_count(train, features=categorical_features, dfs_to_join=None)
    print(train.shape, len(categorical_features))

    # modeling
    clfs, score = modeling_cross_validation(model_params, train[train_features], train['target'],)

    # test
    test = pd.read_csv('../input/test.csv', parse_dates=["first_active_month"])
    test, _ = missing_impute(test, dict_impute)    
    test = datetime_extract(test, dt_col='first_active_month', reference_date=train['first_active_month'].max())

    test = pd.merge(test, df_agg_historical, on="card_id", how="left")
    test = pd.merge(test, df_agg_new_merchant, on="card_id", how="left")
    test = pd.merge(test, df_agg_historical_interact, on="card_id", how="left")
    test, _ = process_count(test, dfs_to_join=counts_dfs)

    # save to
    filename = 'subm_{:.6f}_{}_{}.csv'.format(score, 'LGBM', dt.now().strftime('%Y-%m-%d-%H-%M'))
    print('save to {}'.format(filename))
    get_importances(clfs, feature_names=train_features)
    subm = predict_cross_validation(test[train_features], clfs)
    subm.to_csv(filename, index=False)


if __name__ == '__main__':
    main()

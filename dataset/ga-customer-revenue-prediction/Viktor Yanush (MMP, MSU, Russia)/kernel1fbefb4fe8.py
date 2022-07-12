# author: Viktor Yanush
# Inspired by https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480

import sys
import gc

import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

import functools
from multiprocessing import Pool

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** .5


def read_file(file_name):
    print('Reading {}'.format(file_name))
    return pd.read_csv(
        file_name,
        usecols=['channelGrouping', 'date', 'fullVisitorId', 'sessionId', 'totals', 'device', 'geoNetwork', 'socialEngagementType', 'trafficSource', 'visitStartTime'],
        converters={column: json.loads for column in ['device', 'geoNetwork', 'totals', 'trafficSource']},
        dtype={
            'channelGrouping': str,
            'date': str,
            'fullVisitorId': str,
            'sessionId': str,
        }
    )


def populate_data(df):

    for column in ['device', 'geoNetwork', 'totals', 'trafficSource']:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
    df.drop(const_cols, axis=1, inplace=True)

    if 'campaignCode' in df.columns:
        df.drop('campaignCode', axis=1, inplace=True)
    
    df['date'] = pd.to_datetime(df['visitStartTime'])
    df['sess_date_dow'] = df['date'].dt.dayofweek
    df['sess_date_dom'] = df['date'].dt.day
    df['sess_date_hour'] = df['date'].dt.hour

    try:
        df['transactionRevenue'] = pd.to_numeric(df['transactionRevenue'])
    except:
        df['transactionRevenue'] = pd.Series(np.zeros(df.shape[0]))
    
    df['dummy'] = 1
    df['user_cumcnt_per_day'] = (df[['fullVisitorId','date', 'dummy']].groupby(['fullVisitorId','date'])['dummy'].cumcount()+1)
    df['user_sum_per_day'] = df[['fullVisitorId','date', 'dummy']].groupby(['fullVisitorId','date'])['dummy'].transform(sum)
    df['user_cumcnt_sum_ratio_per_day'] = df['user_cumcnt_per_day'] / df['user_sum_per_day'] 
    df.drop('dummy', axis=1, inplace=True)

    return df


def factorize_categoricals(df, cat_indexers=None):

    # Find categorical features
    cat_feats = [f for f in df.columns
                 if ((df[f].dtype == 'object')
                     & (f not in ['fullVisitorId', 'sessionId', 'date',
                                  'totals', 'device', 'geoNetwork', 'device', 'trafficSource']))]
    print('Categorical features : {}'.format(cat_feats))

    if cat_indexers is None:
        cat_indexers = {}
        for f in cat_feats:
            df[f], indexer = pd.factorize(df[f])
            cat_indexers[f] = indexer
    else:
        for f in cat_feats:
            df[f] = cat_indexers[f].get_indexer(df[f])

    return df, cat_indexers, cat_feats


def aggregate_sessions(df, cat_feats, sum_of_logs=False):
    """
    Aggregate session data for each fullVisitorId
    :param df: DataFrame to aggregate on
    :param cat_feats: List of Categorical features
    :param sum_of_logs: if set to True, revenues are first log transformed and then summed up  
    :return: aggregated fullVisitorId data over Sessions
    """
    if sum_of_logs is True:
        # Go to log first
        df['transactionRevenue'] = np.log1p(df['transactionRevenue'])

    aggs = {}
    for f in cat_feats + ['sess_date_dow', 'sess_date_dom', 'sess_date_hour']:
        aggs[f] = ['min', 'max', 'mean', 'median', 'var', 'std']

    aggs.update({
        'date': ['min', 'max'],
        'transactionRevenue': ['sum', 'size'],
        'hits': ['sum', 'min', 'max', 'mean', 'median'],
        'pageviews': ['sum', 'min', 'max', 'mean', 'median'],
        'bounces': ['sum', 'mean', 'median'],
        'newVisits': ['sum', 'mean', 'median']
    })

    users = df.groupby('fullVisitorId').agg(aggs)
    print('User aggregation done')

    # This may not work in python 3.5, since keys ordered is not guaranteed
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    print('New columns are : {}'.format(new_columns))
    users.columns = new_columns

    # Add dates
    users['date_diff'] = (users.date_max - users.date_min).astype(np.int64) // (24 * 3600 * 1e9)
    
    # Go to log if not already done
    if sum_of_logs is False:
        # Go to log first
        users['transactionRevenue_sum'] = np.log1p(users['transactionRevenue_sum'])

    return users


def get_user_data(file_name='../input/train.csv', cat_indexers=None, sum_of_logs=False):
    data = read_file(file_name=file_name)
    print('Data shape = {}'.format(data.shape))
    
    data = populate_data(df=data)
    data, cat_indexers, cat_feats = factorize_categoricals(df=data, cat_indexers=cat_indexers)
    users = aggregate_sessions(df=data, cat_feats=cat_feats, sum_of_logs=sum_of_logs)

    del data
    gc.collect()

    y = users['transactionRevenue_sum']
    users.drop(['date_min', 'date_max', 'transactionRevenue_sum'], axis=1, inplace=True)

    print('Data shape is now : {}'.format(users.shape))
    return users, y, cat_indexers


def main(sum_of_logs=False):
    trn_users, trn_y, indexers = get_user_data(file_name='../input/train.csv', cat_indexers=None, sum_of_logs=sum_of_logs)
    sub_users, _, _ = get_user_data(file_name='../input/test.csv', cat_indexers=indexers)

    folds = KFold(n_splits=5, shuffle=True, random_state=7956112)

    sub_preds = np.zeros(sub_users.shape[0])
    oof_preds = np.zeros(trn_users.shape[0])
    oof_scores = []
    lgb_params = {
        'learning_rate': 0.03,
        'n_estimators': 3000,
        'num_leaves': 128,
        'subsample': 0.2217,
        'colsample_bytree': 0.6810,
        'min_split_gain': 1e-5,
        'reg_alpha': 5e-4,
        'reg_lambda': 1.3e-5,
        'min_child_weight': 100.,
        'silent': True
    }
    
    for fold_, (trn_, val_) in enumerate(folds.split(trn_users)):
        model = lgb.LGBMRegressor(**lgb_params)

        model.fit(
            trn_users.iloc[trn_], trn_y.iloc[trn_],
            eval_set=[(trn_users.iloc[trn_], trn_y.iloc[trn_]),
                      (trn_users.iloc[val_], trn_y.iloc[val_])],
            eval_metric='rmse',
            early_stopping_rounds=100,
            verbose=0
        )

        oof_preds[val_] = model.predict(trn_users.iloc[val_])
        curr_sub_preds = model.predict(sub_users)
        curr_sub_preds[curr_sub_preds < 0] = 0
        sub_preds += curr_sub_preds / folds.n_splits
        #     preds[preds <0] = 0

        print('Fold %d RMSE (raw output) : %.5f' % (fold_ + 1, rmse(trn_y.iloc[val_], oof_preds[val_])))
        oof_preds[oof_preds < 0] = 0
        oof_scores.append(rmse(trn_y.iloc[val_], oof_preds[val_]))
        print('Fold %d RMSE : %.5f' % (fold_ + 1, oof_scores[-1]))

    print('Full OOF RMSE (zero clipped): %.5f +/- %.5f' % (rmse(trn_y, oof_preds), float(np.std(oof_scores))))

    sub_users['PredictedLogRevenue'] = sub_preds
    sub_users[['PredictedLogRevenue']].to_csv("simple_lgb.csv", index=True)

    print('Submission data shape : {}'.format(sub_users[['PredictedLogRevenue']].shape))


if __name__ == '__main__':
    gc.enable()
    main(sum_of_logs=False)

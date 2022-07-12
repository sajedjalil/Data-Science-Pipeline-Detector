import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import functools
from multiprocessing import Pool
import logging
import gc
import matplotlib.pyplot as plt
import sys


TRN_PATH = '../input/train.csv'
SUB_PATH = '../input/test.csv'


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** .5


def get_dict_content(dct_str=None, key='visits'):
    try:
        return float(eval(dct_str)[key])
    except KeyError:
        return 0.0


def get_dict_content_str(dct_str=None, key='visits'):
    try:
        return eval(dct_str)[key]
    except NameError:
        return eval(dct_str.replace('false', 'False').replace('true', 'True'))[key]
    except KeyError:
        return np.nan


def apply_func_on_series(data=None, func=None, key=None):
    return data.apply(lambda x: func(x, key=key))


def multi_apply_func_on_series(df=None, func=None, key=None, n_jobs=4):
    p = Pool(n_jobs)
    f_ = p.map(functools.partial(apply_func_on_series, func=func, key=key),
               np.array_split(df, n_jobs))
    f_ = pd.concat(f_, axis=0, ignore_index=True)
    p.close()
    p.join()
    return f_.values


def read_file(file_name=None, nrows=None):
    logger.info('Reading {}'.format(file_name))
    return pd.read_csv(
        file_name,
        usecols=['channelGrouping', 'date', 'fullVisitorId', 'sessionId', 'totals', 'device', 'geoNetwork', 'socialEngagementType', 'trafficSource', 'visitStartTime'],
        dtype={
            'channelGrouping': str,
            'geoNetwork': str,
            'date': str,
            'fullVisitorId': str,
            'sessionId': str,
            'totals': str,
            'device': str,
        },
        nrows=nrows,  # 50000
    )


def populate_data(df=None):

    # Add data features
    df['date'] = pd.to_datetime(df['visitStartTime'])
    df['sess_date_dow'] = df['date'].dt.dayofweek
    df['sess_date_dom'] = df['date'].dt.day
    df['sess_date_hour'] = df['date'].dt.hour
    # df['sess_date_week'] = df['date'].dt.weekofyear

    for f in ['transactionRevenue', 'visits', 'hits', 'pageviews', 'bounces', 'newVisits']:
        df[f] = multi_apply_func_on_series(
            df=df['totals'],
            func=get_dict_content,
            key=f,
            n_jobs=4
        )
        logger.info('Done with totals.{}'.format(f))

    for f in ['continent', 'subContinent', 'country', 'region', 'metro', 'city', 'networkDomain']:
        df[f] = multi_apply_func_on_series(
            df=df['geoNetwork'],
            func=get_dict_content_str,
            key=f,
            n_jobs=4
        )
        logger.info('Done with geoNetwork.{}'.format(f))

    for f in ['browser', 'operatingSystem', 'isMobile', 'deviceCategory']:
        df[f] = multi_apply_func_on_series(
            df=df['device'],
            func=get_dict_content_str,
            key=f,
            n_jobs=4
        )
        logger.info('Done with device.{}'.format(f))
        
    for f in ['source', 'medium']:
        df[f] = multi_apply_func_on_series(
            df=df['trafficSource'],
            func=get_dict_content_str,
            key=f,
            n_jobs=4
        )
        logger.info('Done with trafficSource.{}'.format(f))
        

    df.drop(['totals', 'geoNetwork', 'device', 'trafficSource', 'visitStartTime'], axis=1, inplace=True)
    
    # This is all Scirpus' fault :) https://www.kaggle.com/scirpus
    # 
    df['dummy'] = 1
    df['user_cumcnt_per_day'] = (df[['fullVisitorId','date', 'dummy']].groupby(['fullVisitorId','date'])['dummy'].cumcount()+1)
    df['user_sum_per_day'] = df[['fullVisitorId','date', 'dummy']].groupby(['fullVisitorId','date'])['dummy'].transform(sum)
    df['user_cumcnt_sum_ratio_per_day'] = df['user_cumcnt_per_day'] / df['user_sum_per_day'] 
    df.drop('dummy', axis=1, inplace=True)

    return df


def factorize_categoricals(df=None, cat_indexers=None):

    # Find categorical features
    cat_feats = [f for f in df.columns
                 if ((df[f].dtype == 'object')
                     & (f not in ['fullVisitorId', 'sessionId', 'date',
                                  'totals', 'device', 'geoNetwork', 'device', 'trafficSource']))]
    logger.info('Categorical features : {}'.format(cat_feats))

    if cat_indexers is None:
        cat_indexers = {}
        for f in cat_feats:
            df[f], indexer = pd.factorize(df[f])
            cat_indexers[f] = indexer
    else:
        for f in cat_feats:
            df[f] = cat_indexers[f].get_indexer(df[f])

    return df, cat_indexers, cat_feats


def aggregate_sessions(df=None, cat_feats=None, sum_of_logs=False):
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

    aggs = {
        'date': ['min', 'max'],
        'transactionRevenue': ['sum', 'size'],
        'hits': ['sum', 'min', 'max', 'mean', 'median'],
        'pageviews': ['sum', 'min', 'max', 'mean', 'median'],
        'bounces': ['sum', 'mean', 'median'],
        'newVisits': ['sum', 'mean', 'median']
    }

    for f in cat_feats + ['sess_date_dow', 'sess_date_dom', 'sess_date_hour']:
        aggs[f] = ['min', 'max', 'mean', 'median', 'var', 'std']

    users = df.groupby('fullVisitorId').agg(aggs)
    logger.info('User aggregation done')

    # This may not work in python 3.5, since keys ordered is not guaranteed
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    logger.info('New columns are : {}'.format(new_columns))
    users.columns = new_columns

    # Add dates
    users['date_diff'] = (users.date_max - users.date_min).astype(np.int64) // (24 * 3600 * 1e9)
    
    # Go to log if not already done
    if sum_of_logs is False:
        # Go to log first
        users['transactionRevenue_sum'] = np.log1p(users['transactionRevenue_sum'])

    return users


def get_user_data(file_name='../input/train.csv', cat_indexers=None, nrows=None, sum_of_logs=False):

    data = read_file(file_name=file_name, nrows=nrows)
    logger.info('Data shape = {}'.format(data.shape))
    
    data = populate_data(df=data)

    data, cat_indexers, cat_feats = factorize_categoricals(df=data, cat_indexers=cat_indexers)

    users = aggregate_sessions(df=data, cat_feats=cat_feats, sum_of_logs=sum_of_logs)

    del data
    gc.collect()

    y = users['transactionRevenue_sum']
    users.drop(['date_min', 'date_max', 'transactionRevenue_sum'], axis=1, inplace=True)

    logger.info('Data shape is now : {}'.format(users.shape))

    return users, y, cat_indexers


def main(sum_of_logs=False, nrows=None):
    try:
        trn_users, trn_y, indexers = get_user_data(file_name=TRN_PATH, cat_indexers=None, nrows=nrows, sum_of_logs=sum_of_logs)
        sub_users, _, _ = get_user_data(file_name=SUB_PATH, cat_indexers=indexers, nrows=nrows)
    
        folds = KFold(n_splits=5, shuffle=True, random_state=7956112)
    
        sub_preds = np.zeros(sub_users.shape[0])
        oof_preds = np.zeros(trn_users.shape[0])
        oof_scores = []
        lgb_params = {
            'learning_rate': 0.03,
            'n_estimators': 2000,
            'num_leaves': 128,
            'subsample': 0.2217,
            'colsample_bytree': 0.6810,
            'min_split_gain': np.power(10.0, -4.9380),
            'reg_alpha': np.power(10.0, -3.2454),
            'reg_lambda': np.power(10.0, -4.8571),
            'min_child_weight': np.power(10.0, 2),
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
    
            logger.info('Fold %d RMSE (raw output) : %.5f' % (fold_ + 1, rmse(trn_y.iloc[val_], oof_preds[val_])))
            oof_preds[oof_preds < 0] = 0
            oof_scores.append(rmse(trn_y.iloc[val_], oof_preds[val_]))
            logger.info('Fold %d RMSE : %.5f' % (fold_ + 1, oof_scores[-1]))
    
        logger.info('Full OOF RMSE (zero clipped): %.5f +/- %.5f' % (rmse(trn_y, oof_preds), float(np.std(oof_scores))))
    
        # Stay in logs for submission
        sub_users['PredictedLogRevenue'] = sub_preds
        sub_users[['PredictedLogRevenue']].to_csv("simple_lgb.csv", index=True)
    
        logger.info('Submission data shape : {}'.format(sub_users[['PredictedLogRevenue']].shape))
    
        hist, bin_edges = np.histogram(np.hstack((oof_preds, sub_preds)), bins=25)
        plt.figure(figsize=(12, 7))
        plt.title('Distributions of OOF and TEST predictions', fontsize=15, fontweight='bold')
        plt.hist(oof_preds, label='OOF predictions', alpha=.6, bins=bin_edges, density=True, log=True)
        plt.hist(sub_preds, label='TEST predictions', alpha=.6, bins=bin_edges, density=True, log=True)
        plt.legend()
        plt.savefig('distributions.png')
        
    except Exception as err:
        logger.exception("Unexpected error")
        

def get_logger():
    logger_ = logging.getLogger('main')
    logger_.setLevel(logging.DEBUG)
    fh = logging.FileHandler('simple_lightgbm.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]%(asctime)s:%(name)s:%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger_.addHandler(fh)
    logger_.addHandler(ch)

    return logger_


if __name__ == '__main__':
    gc.enable()
    logger = get_logger()
    main(sum_of_logs=False, nrows=None)

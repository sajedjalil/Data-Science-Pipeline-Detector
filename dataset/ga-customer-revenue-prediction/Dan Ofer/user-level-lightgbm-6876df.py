import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import functools
from multiprocessing import Pool
import logging
import gc


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
        usecols=['channelGrouping', 'date', 'fullVisitorId', 'sessionId', 'totals', 'device', 'geoNetwork'],
        dtype={
            'channelGrouping': str,
            'geoNetwork': str,
            'date': str,
            'fullVisitorId': str,
            'sessionId': str,
            'totals': str,
            'device': str
        },
        nrows=nrows,  # 50000
    )


def populate_data(df=None):

    # Add data features
    df['date'] = pd.to_datetime(df['date'])
    df['sess_date_dow'] = df['date'].dt.dayofweek
    # df['sess_date_doy'] = df['date'].dt.dayofyear
    # df['sess_date_mon'] = df['date'].dt.month
    df['sess_date_week'] = df['date'].dt.weekofyear

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

    df.drop(['totals', 'geoNetwork', 'device'], axis=1, inplace=True)

    return df


def factorize_categoricals(df=None, cat_indexers=None):

    # Find categorical features
    cat_feats = [f for f in df.columns
                 if ((df[f].dtype == 'object')
                     & (f not in ['fullVisitorId', 'sessionId', 'date',
                                  'totals', 'device', 'geoNetwork', 'device']))]
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


def aggregate_sessions(df=None, cat_feats=None):

    aggs = {
        'date': ['min', 'max'],
        'transactionRevenue': ['sum'],
        'hits': ['sum', 'min', 'max', 'mean', 'median',"last"],
        'pageviews': ['sum', 'min', 'max', 'mean', 'size',"last"],
        'bounces': ['sum', 'max',"last"],
        'newVisits': ['sum', 'size',"last"]
    }

    for f in cat_feats: 
        #+ ['sess_date_dow', 'sess_date_doy', 'sess_date_mon', 'sess_date_week']:
        aggs[f] = ['min', 'max', 'mean']

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

    return users


def get_user_data(file_name='../input/train.csv', cat_indexers=None, nrows=None):

    data = read_file(file_name=file_name, nrows=nrows)
    logger.info('Data shape = {}'.format(data.shape))
    
    data = populate_data(df=data)

    data, cat_indexers, cat_feats = factorize_categoricals(df=data, cat_indexers=cat_indexers)

    users = aggregate_sessions(df=data, cat_feats=cat_feats)

    del data
    gc.collect()

    y = users['transactionRevenue_sum']
    # users.drop(['date_min', 'date_max', 'transactionRevenue_sum'], axis=1, inplace=True)

    logger.info('Data shape is now : {}'.format(users.shape))

    return users, y, cat_indexers


def main(nrows=None):
    trn_users, trn_y, indexers = get_user_data(file_name=TRN_PATH, cat_indexers=None, nrows=nrows)
    print("Train features extracted")
    trn_users.to_csv("train_galytics_agg_v2.csv.gz",index=False,compression="gzip")
    print("Train saved")
    sub_users, _, _ = get_user_data(file_name=SUB_PATH, cat_indexers=indexers, nrows=nrows)
    print("test features extracted")
    sub_users.to_csv("test_galytics_agg_v2.csv.gz",index=False,compression="gzip")
    
    
    # folds = KFold(n_splits=5, shuffle=True, random_state=7956112)

    # sub_preds = np.zeros(sub_users.shape[0])
    # oof_preds = np.zeros(trn_users.shape[0])
    # oof_scores = []
    # for fold_, (trn_, val_) in enumerate(folds.split(trn_users)):
    #     model = lgb.LGBMRegressor()

    #     model.fit(
    #         trn_users.iloc[trn_], np.log1p(trn_y.iloc[trn_]),
    #         eval_set=[(trn_users.iloc[trn_], np.log1p(trn_y.iloc[trn_])),
    #                   (trn_users.iloc[val_], np.log1p(trn_y.iloc[val_]))],
    #         eval_metric='rmse',
    #         verbose=10
    #     )

    #     oof_preds[val_] = model.predict(trn_users.iloc[val_])
    #     curr_sub_preds = model.predict(sub_users)
    #     curr_sub_preds[curr_sub_preds < 0] = 0
    #     sub_preds += curr_sub_preds / folds.n_splits
    #     #     preds[preds <0] = 0

    #     logger.info('Fold %d RMSE : %.5f' % (fold_ + 1, rmse(np.log1p(trn_y.iloc[val_]), oof_preds[val_])))
    #     oof_preds[oof_preds < 0] = 0
    #     oof_scores.append(rmse(np.log1p(trn_y.iloc[val_]), oof_preds[val_]))
    #     logger.info('Fold %d RMSE : %.5f' % (fold_ + 1, oof_scores[-1]))

    # logger.info('Full OOF RMSE : %.5f +/- %.5f' % (rmse(np.log1p(trn_y), oof_preds), float(np.std(oof_scores))))

    # # Stay in logs for submission
    # sub_users['PredictedLogRevenue'] = sub_preds
    # sub_users[['PredictedLogRevenue']].to_csv("simple_lgb.csv", index=True)

    # logger.info('Submission data shape : {}'.format(sub_users[['PredictedLogRevenue']].shape))


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
    main(nrows=None)  # 50000)

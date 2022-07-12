import gc
import json
import os
import re
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from scipy.sparse import hstack
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

id_col = 'fullVisitorId'


@contextmanager
def timer(name):
    print(f'【{name}】 begin at 【{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}】')
    t0 = time.time()
    yield
    print(f'【{name}】 done in 【{time.time() - t0:.0f}】 s')


def get_time_stamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def flatten_counter(cnts):
    pairs = sorted(cnts.items(), key=lambda pair: (pair[1], pair[0]))
    last_cnt = 0
    new_pairs = []
    for k, v in pairs:
        if v > last_cnt:
            new_pairs.append((k, v))
            last_cnt = v
        else:
            last_cnt += 1
            new_pairs.append((k, last_cnt))
    return dict(new_pairs)


class CountEncoder:
    def __init__(self, nan_value=-127, count_unique=True):
        self.counter = None
        self.nan_value = nan_value
        self.unseen_code_value = nan_value
        self.count_unique = count_unique

    def fit(self, x, need_fill_na=True):
        x = x.fillna(self.nan_value) if need_fill_na else x
        self.counter = Counter(x)
        unseen_code = None
        if not x.loc[x == self.nan_value].shape[0]:
            unseen_code = sorted(self.counter.items(), key=lambda pair: (pair[1], pair[0]), reverse=True)[0][0]
        self.counter = flatten_counter(self.counter) if self.count_unique else self.counter
        if unseen_code is not None:
            self.unseen_code_value = self.counter[unseen_code]

    def transform(self, x, need_fill_na=True):
        x = x.fillna(self.nan_value) if need_fill_na else x
        return x.apply(lambda ele: self.unseen_code_value if (
            ele == self.nan_value or ele not in self.counter) else self.counter[ele])

    def fit_transform(self, x):
        x = x.fillna(self.nan_value)
        self.fit(x, need_fill_na=False)
        return self.transform(x, need_fill_na=False)


def get_int_type(col):
    max_val = col.max()
    if max_val < 2 ** 7:
        return np.int8
    elif max_val < 2 ** 15:
        return np.int16
    elif max_val < 2 ** 31:
        return np.int32
    else:
        return np.int64


def encode_obj(df, cols=None, encoders=None, data_types=None, count_unique=True):
    if not encoders:
        encoders = {}
        data_types = {}
        for col in np.intersect1d(df.columns, cols):
            cer = CountEncoder(count_unique=count_unique)
            df[col] = cer.fit_transform(df[col])
            encoders[col] = cer
            data_type = get_int_type(df[col])
            data_types[col] = data_type
            df[col] = df[col].astype(data_type)
    else:
        for col in np.intersect1d(df.columns, cols):
            df[col] = encoders[col].transform(df[col]).astype(data_types[col])

    return df, encoders, data_types


def get_ids():
    cols = [id_col, 'visitStartTime']

    with timer('get sub_ids'):
        test_df = pd.read_csv('../input/test_v2.csv', dtype={id_col: 'str'}).loc[:, cols]
        sub_df = pd.read_csv('../input/sample_submission_v2.csv', dtype={id_col: 'str'})
        gc.collect()
        print(f'test_df: {test_df.shape}, sub_df: {sub_df.shape}')
        s = test_df[id_col].unique()
        sub_ids = sub_df[id_col].copy()
        u_sub_ids = sub_ids.unique()
        print(f'ids@test_df: {s.shape}, sub_ids: {sub_ids.shape}, u_sub_ids: {u_sub_ids.shape}')
        c_sub_ids = np.intersect1d(s, u_sub_ids)
        del sub_df, s, u_sub_ids
        gc.collect()
        print(f'c_sub_ids: {c_sub_ids.shape}')

    with timer('get tr_ids'):
        def read_cols(file_tag, _cols):
            data_reader = pd.read_csv(f'../input/{file_tag}_v2.csv', iterator=True, chunksize=20000,
                                      dtype={id_col: 'str'})
            _df = pd.DataFrame()
            for it in data_reader:
                _df = _df.append(it[_cols].copy(), ignore_index=True)
                del it
                gc.collect()
            del data_reader
            gc.collect()
            return _df

        train_df = read_cols('train', cols)
        gc.collect()
        print(f'train_df: {train_df.shape}')
        df = train_df.append(test_df, ignore_index=True)
        gc.collect()
        print(f'df: {df.shape}')

        cnt = df[id_col].value_counts()
        tr_ids = cnt.loc[cnt > 1].index.values
        tr_df = df.loc[df[id_col].isin(tr_ids)]
        gc.collect()
        print(f'ids: {cnt.shape}, tr_ids(cnt>1): {tr_ids.shape}, tr_df(cnt>1): {tr_df.shape}')

        tm_col = 'visitStartTime'
        time_frame_start1 = pd.to_datetime('2017-05-01').timestamp()
        time_frame_end1 = pd.to_datetime('2017-10-01').timestamp()
        time_frame_start2 = pd.to_datetime('2018-01-01').timestamp()
        time_frame_end2 = pd.to_datetime('2018-06-01').timestamp()

        tr_ids1 = cnt.loc[1 == cnt].index.values
        tr_df1 = df.loc[df[id_col].isin(tr_ids1) & (df[tm_col] >= time_frame_start1) & (df[tm_col] < time_frame_end1)]
        tr_df2 = df.loc[df[id_col].isin(tr_ids1) & (df[tm_col] >= time_frame_start2) & (df[tm_col] < time_frame_end2)]
        print(f'tr_ids1(1==cnt): {tr_ids1.shape}, tr_df1(frame1): {tr_df1.shape}, tr_df2(frame2): {tr_df2.shape}')

        tr_ids1 = tr_df1[id_col].unique()
        tr_ids2 = tr_df2[id_col].unique()
        tr_ids = reduce(np.union1d, (tr_ids, tr_ids1, tr_ids2))
        print(f'tr_ids1(frame1): {tr_ids1.shape}, tr_ids2(frame2): {tr_ids2.shape}')
        df = df.loc[df[id_col].isin(tr_ids)]
        gc.collect()
        print(f'tr_ids: {tr_ids.shape}, df: {df.shape}')
        del df, train_df, test_df, tr_df, cnt, tr_ids1, tr_ids2, tr_df1, tr_df2
        gc.collect()

    return tr_ids, sub_ids


def extract_hits(df):
    def _extract_hits(rec):
        target = rec['target']
        info = re.sub(r"\\?['\"]", "'", rec['hits'])

        meta_infos = re.findall(r"'time':\s*'(\d+)'.*?exitScreenName.*?/([^/]+?)',", info)
        times, exitScreenNames = zip(*meta_infos) if meta_infos else ([], [])
        times = [int(t) for t in times] if times else []
        exitScreenNames = ' '.join(set(exitScreenNames)) if exitScreenNames else 'None'

        promoNames = set(re.findall(r"promoName':\s*'(.+?)',", info))
        promoNames = ' '.join(promoNames) if promoNames else 'None'

        products = re.findall(
            "v2ProductName':\s*'(.+?)',.*?v2ProductCategory':\s*'(.+?)',.*?'productPrice':\s*'(\d+)',", info)
        productNames, productCategorys, productPrices = zip(*products) if products else ([], [], [])
        productPrices = [int(price) for price in productPrices] if productPrices else []
        products = dict(zip(productNames, productPrices))
        productNames = ' '.join(products.keys()) if products else 'None'
        productPrices = list(products.values())
        productCategorys = ' '.join(set(productCategorys)) if products else 'None'
        hitProductNames = [name for name, price in products.items() if
                           price == target] if target > 0 and products else []
        hitProductNames = ' '.join(hitProductNames) if hitProductNames else 'None'

        hit_cnt = len(times)
        total_time = np.max(times) if times else 0
        mean_time = total_time // (hit_cnt - 1) if hit_cnt > 1 else 0
        diff_times = np.diff(times)
        mean_diff_time, min_diff_time, max_diff_time, std_diff_time = 0, 0, 0, 0
        if diff_times.shape[0] > 0:
            mean_diff_time = int(np.mean(diff_times))
            min_diff_time = np.min(diff_times)
            max_diff_time = np.max(diff_times)
            std_diff_time = int(np.std(diff_times))

        mean_price = int(np.mean(productPrices)) if productPrices else 0
        min_price = np.min(productPrices) if productPrices else 0
        max_price = np.max(productPrices) if productPrices else 0
        std_price = int(np.std(productPrices)) if productPrices else 0
        mean_price_ratio = target / mean_price if mean_price else -1.0
        min_price_ratio = target / min_price if min_price else -1.0
        max_price_ratio = target / max_price if max_price else -1.0

        return (total_time, mean_time, mean_diff_time, min_diff_time, max_diff_time, std_diff_time, exitScreenNames,
                promoNames, productNames, productCategorys, mean_price, min_price, max_price, std_price,
                mean_price_ratio, min_price_ratio, max_price_ratio, hitProductNames)

    (df['total_time'], df['mean_time'], df['mean_diff_time'], df['min_diff_time'], df['max_diff_time'],
     df['std_diff_time'], df['exitScreenNames_src'], df['promoNames_src'], df['productNames_src'],
     df['productCategorys_src'], df['mean_price'], df['min_price'], df['max_price'], df['std_price'],
     df['mean_price_ratio'], df['min_price_ratio'], df['max_price_ratio'], df['hitProductNames_src']) = zip(
        *df[['hits', 'target']].apply(_extract_hits, axis=1))

    return df.drop('hits', axis=1)


def load_df(ids, data_tag='train'):
    json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']

    data_reader = pd.read_csv(f'../input/{data_tag}_v2.csv', iterator=True, chunksize=100000,
                              converters={column: json.loads for column in json_cols}, dtype={id_col: 'str'})
    df = pd.DataFrame()
    for data in data_reader:
        data = data.loc[data[id_col].isin(ids)].reset_index(drop=True).copy()
        if data.shape[0] > 0:
            print(f'before: {data.shape}', end='\t')
            for column in json_cols:
                column_as_df = json_normalize(data[column])
                column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
                data = data.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
                del column_as_df
                gc.collect()

            data = data.rename(columns={'totals.transactionRevenue': 'target'})
            data['target'] = data['target'].fillna(0) if 'target' in data.columns else 0
            data.target = data.target.astype(np.int64)
            gc.collect()
            data = extract_hits(data)
            gc.collect()

            df = df.append(data, ignore_index=True, sort=False)
            print(f'after: {data.shape}, total: {df.shape}')
            del data
            gc.collect()
    return df


def get_record_data(tr_ids, sub_ids):
    with timer('load train data'):
        train_df_r = load_df(tr_ids)
        print(f'train_df_r: {train_df_r.shape}')
        gc.collect()
        test_df = load_df(tr_ids, 'test')
        print(f'test_df: {test_df.shape}')
        gc.collect()
        train_df_r = train_df_r.append(test_df, ignore_index=True, sort=False)
        del test_df
        gc.collect()
        print(f'train_df_r: {train_df_r.shape}')

    with timer('drop still cols'):
        still_cols = ['trafficSource.adwordsClickInfo.gclId']
        for col in train_df_r.columns:
            cnt = train_df_r[col].nunique(dropna=False)
            if cnt <= 1:
                still_cols.append(col)
        train_df_r = train_df_r.drop(still_cols, axis=1)
        gc.collect()
        print(f'train_df_r: {train_df_r.shape}')
        print(f'still_cols({len(still_cols)}): {still_cols}')

    with timer('stat pv'):
        def encode_user_pv(df, gp_col=id_col, col_prefix='user', sort_col='visitStartTime',
                           session_interval=7 * 24 * 3600, window_interval=60 * 24 * 3600):
            # global scope
            df = df.sort_values(by=[gp_col, sort_col])
            gp = df.groupby(gp_col)[sort_col]
            df[f'{col_prefix}_g_span'] = gp.diff().fillna(0)

            # window scope
            df = df.join(gp.max(), on=gp_col, rsuffix='_rm').rename(columns={f'{sort_col}_rm': 'window_id'})
            df['window_id'] = ((df['window_id'] - df[sort_col]) // window_interval).astype(np.uint8)
            gp_cols = [gp_col, 'window_id']
            gp = df.groupby(gp_cols)
            df = df.join(gp[sort_col].count(), on=gp_cols, rsuffix='_w_pv').rename(
                columns={f'{sort_col}_w_pv': f'{col_prefix}_w_pv'})
            df = df.join(gp[sort_col].max() - gp[sort_col].min(), on=gp_cols, rsuffix='_w_span').rename(
                columns={f'{sort_col}_w_span': f'{col_prefix}_w_span'})
            df = df.join(gp[f'{col_prefix}_g_span'].first(), on=gp_cols, rsuffix='_w_idle').rename(
                columns={f'{col_prefix}_g_span_w_idle': f'{col_prefix}_w_idle'})

            # session scope
            df['session_id'] = df[f'{col_prefix}_g_span']
            df.loc[df['session_id'] <= session_interval, 'session_id'] = 0
            df.loc[df['session_id'] > session_interval, 'session_id'] = 1
            df['session_id'] = df.groupby(gp_col)['session_id'].cumsum().astype(np.uint16)
            gp_cols = [gp_col, 'session_id']
            gp = df.groupby(gp_cols)
            df = df.join(gp[sort_col].count(), on=gp_cols, rsuffix='_s_pv').rename(
                columns={f'{sort_col}_s_pv': f'{col_prefix}_s_pv'})
            df = df.join(gp[sort_col].max() - gp[sort_col].min(), on=gp_cols, rsuffix='_s_span').rename(
                columns={f'{sort_col}_s_span': f'{col_prefix}_s_span'})
            df = df.join(gp[f'{col_prefix}_g_span'].first(), on=gp_cols, rsuffix='_s_idle').rename(
                columns={f'{col_prefix}_g_span_s_idle': f'{col_prefix}_s_idle'})

            # visit scope
            gp_cols = [gp_col, 'visitId']
            gp = df.groupby(gp_cols)
            df = df.join(gp[sort_col].count(), on=gp_cols, rsuffix='_v_pv').rename(
                columns={f'{sort_col}_v_pv': f'{col_prefix}_v_pv'})
            df = df.join(gp[sort_col].max() - gp[sort_col].min(), on=gp_cols, rsuffix='_v_span').rename(
                columns={f'{sort_col}_v_span': f'{col_prefix}_v_span'})
            df = df.join(gp[f'{col_prefix}_g_span'].first(), on=gp_cols, rsuffix='_v_idle').rename(
                columns={f'{col_prefix}_g_span_v_idle': f'{col_prefix}_v_idle'})

            for flag in list('gwsv'):
                df[f'{col_prefix}_{flag}_span'] = df[f'{col_prefix}_{flag}_span'].astype(np.int32)
            for flag in list('wsv'):
                df[f'{col_prefix}_{flag}_pv'] = df[f'{col_prefix}_{flag}_pv'].astype(np.int16)
                df[f'{col_prefix}_{flag}_idle'] = df[f'{col_prefix}_{flag}_idle'].astype(np.int32)
            df = df.sort_index()

            return df

        train_df_r = encode_user_pv(train_df_r)
        gc.collect()
        print(f'train_df_r: {train_df_r.shape}')

    with timer('encode objs'):
        obj_cols = ['channelGrouping', 'customDimensions', 'device.browser', 'device.deviceCategory', 'device.isMobile',
                    'device.operatingSystem', 'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country',
                    'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region', 'geoNetwork.subContinent',
                    'totals.bounces', 'totals.newVisits', 'trafficSource.adContent',
                    'trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adwordsClickInfo.isVideoAd',
                    'trafficSource.adwordsClickInfo.slot', 'trafficSource.campaign', 'trafficSource.isTrueDirect',
                    'trafficSource.keyword', 'trafficSource.medium', 'trafficSource.referralPath',
                    'trafficSource.source']
        col = 'trafficSource.keyword'
        train_df_r[col] = train_df_r[col].str.lower()
        for col in ['device.browser', 'trafficSource.adContent', 'trafficSource.keyword', 'trafficSource.referralPath',
                    'trafficSource.source']:
            train_df_r[f'{col}_src'] = train_df_r[col].fillna('unknown')

        train_df_r, cat_encoders, cat_data_types = encode_obj(train_df_r, obj_cols)
        gc.collect()
        print(f'train_df.shape: {train_df_r.shape}')

    with timer("convert numeric col's data_type"):
        def convert_data_type(df):
            int8_cols = ['totals.sessionQualityDim', 'totals.transactions', 'trafficSource.adwordsClickInfo.page']
            for _col in int8_cols:
                df[_col] = df[_col].fillna(-127).astype(np.int8)

            _col = 'totals.pageviews'
            df[_col] = df[_col].fillna(1)
            int16_cols = ['visitNumber', 'totals.pageviews', 'totals.hits']
            for _col in int16_cols:
                df[_col] = df[_col].fillna(-127).astype(np.int16)

            df.loc[df['totals.timeOnSite'].isnull() & (df['total_time'] > 0), 'totals.timeOnSite'] = np.round(
                df.loc[df['totals.timeOnSite'].isnull() & (df['total_time'] > 0), 'total_time'] / 1000)
            df = df.drop('total_time', axis=1)
            gc.collect()
            int32_cols = ['visitId', 'visitStartTime', 'totals.timeOnSite', 'mean_time', 'mean_diff_time',
                          'min_diff_time', 'max_diff_time', 'std_diff_time']
            for _col in int32_cols:
                df[_col] = df[_col].fillna(-127).astype(np.int32)

            float32_cols = ['mean_price_ratio', 'min_price_ratio', 'max_price_ratio']
            for _col in float32_cols:
                df[_col] = df[_col].fillna(-127).astype(np.float32)

            _col = 'totals.totalTransactionRevenue'
            df[_col] = df[_col].fillna(0).astype(np.int64)
            df['totals.extraRevenue'] = df[_col] - df['target']

            return df

        train_df_r = convert_data_type(train_df_r)
        gc.collect()
        print(f'train_df_r: {train_df_r.shape}')

    with timer('encode trafficSource.referralPath_src'):
        def encode_ref_path(_df, _ref_encoders=None, _ref_data_types=None):
            _col = 'trafficSource.referralPath_src'

            _df['ref_level_cnt'] = _df[_col].str.count('/').astype(np.int8)
            s = _df[_col].str.split('/')
            _df['ref_level_1'] = '/' + s.str.get(1)
            for i in range(2, 5):
                _df[f'ref_level_{i}'] = _df[f'ref_level_{i-1}'] + '/' + s.str.get(i)
            _df['ref_level_n'] = s.str.get(-1)

            _cols = ['ref_level_n'] + [f'ref_level_{i}' for i in range(1, 5)]
            if not _ref_encoders:
                _df, _ref_encoders, _ref_data_types = encode_obj(_df, _cols)
            else:
                _df, _, _ = encode_obj(_df, _cols, _ref_encoders, _ref_data_types)

            return _df, _ref_encoders, _ref_data_types

        train_df_r, ref_encoders, ref_data_types = encode_ref_path(train_df_r)
        gc.collect()
        print(f'train_df_r: {train_df_r.shape}')

    with timer('encode trafficSource.source_src'):
        def encode_source(_df, _src_encoders=None, _src_data_types=None):
            _col = 'trafficSource.source_src'
            _df[_col] = _df[_col].str.replace(r':\d+\s*$', '')

            _df['src_level_cnt'] = _df[_col].str.count('\.').astype(np.int8) + 1
            s = _df[_col].str.split('\.')
            for i in range(1, 4):
                _df[f'src_level_{i}'] = s.str.get(-i)
            _df['src_level_n'] = s.str.get(0)

            _cols = ['src_level_n'] + [f'src_level_{i}' for i in range(1, 4)]
            if not _src_encoders:
                _df, _src_encoders, _src_data_types = encode_obj(_df, _cols)
            else:
                _df, _, _ = encode_obj(_df, _cols, _src_encoders, _src_data_types)

            return _df, _src_encoders, _src_data_types

        train_df_r, src_encoders, src_data_types = encode_source(train_df_r)
        gc.collect()
        print(f'train_df_r: {train_df_r.shape}')

    with timer('encode dates'):
        def encode_dates(df, _date_encoders=None, _date_data_types=None):
            s = pd.to_datetime(df.date.astype(np.str))
            s1 = pd.to_datetime(df.visitStartTime, unit='s')

            df['visit_idle'] = (s1 - s).dt.days.astype(np.int8)
            df['visit_year'] = (2018 - s1.dt.year).astype(np.int8)
            df['visit_month'] = s1.dt.month.astype(np.int8)
            df['visit_day'] = s1.dt.day.astype(np.int8)
            df['visit_hour'] = s1.dt.hour.astype(np.int8)
            df['visit_week'] = s1.dt.week.astype(np.int8)
            df['visit_dayofweek'] = s1.dt.dayofweek.astype(np.int8)
            df['visit_dayofyear'] = s1.dt.dayofyear.astype(np.int16)
            df['visit_quarter'] = s1.dt.quarter.astype(np.int8)
            df['visit_weekend'] = df['visit_dayofweek'].isin([5, 6]).astype(np.int8)

            df['visit_hour_pv'] = df.visitStartTime // 3600
            df['visit_day_pv'] = df.visit_hour_pv // 24
            df, _date_encoders, _date_data_types = encode_obj(df, [f'visit_{tag}_pv' for tag in ['hour', 'day']],
                                                              _date_encoders, _date_data_types, count_unique=False)

            return df.drop('date', axis=1), _date_encoders, _date_data_types

        train_df_r, date_encoders, date_data_types = encode_dates(train_df_r)
        gc.collect()
        train_df_r['visit_delay'] = (train_df_r.visitStartTime - train_df_r.visitId).astype(np.int16)
        gc.collect()
        print(f'train_df_r: {train_df_r.shape}')

    with timer('save train data'):
        train_df_r.to_pickle('train_df_r', compression='gzip')
        del train_df_r
        gc.collect()

    with timer('load test data'):
        tr_df = load_df(sub_ids)
        print(f'tr_df: {tr_df.shape}')
        gc.collect()
        test_df_r = load_df(sub_ids, 'test')
        print(f'test_df_r: {test_df_r.shape}')
        gc.collect()
        test_df_r = tr_df.append(test_df_r, ignore_index=True, sort=False)
        del tr_df
        gc.collect()
        print(f'test_df_r: {test_df_r.shape}')

    with timer('drop still cols'):
        test_df_r = test_df_r.drop(still_cols, axis=1)
        gc.collect()
        print(f'test_df_r: {test_df_r.shape}')

    with timer('stat pv'):
        test_df_r = encode_user_pv(test_df_r)
        gc.collect()
        print(f'test_df_r: {test_df_r.shape}')

    with timer('encode objs'):
        col = 'trafficSource.keyword'
        test_df_r[col] = test_df_r[col].str.lower()
        for col in ['device.browser', 'trafficSource.adContent', 'trafficSource.keyword', 'trafficSource.referralPath',
                    'trafficSource.source']:
            test_df_r[f'{col}_src'] = test_df_r[col].fillna('unknown')

        test_df_r, cat_encoders, cat_data_types = encode_obj(test_df_r, obj_cols, cat_encoders, cat_data_types)
        gc.collect()
        test_df_r = convert_data_type(test_df_r)
        gc.collect()
        print(f'test_df_r: {test_df_r.shape}')

    with timer('encode texts'):
        test_df_r, ref_encoders, ref_data_types = encode_ref_path(test_df_r, ref_encoders, ref_data_types)
        gc.collect()
        test_df_r, src_encoders, src_data_types = encode_source(test_df_r, src_encoders, src_data_types)
        gc.collect()
        print(f'test_df_r: {test_df_r.shape}')

    with timer('encode dates'):
        test_df_r, date_encoders, date_data_types = encode_dates(test_df_r, date_encoders, date_data_types)
        gc.collect()
        test_df_r['visit_delay'] = (test_df_r.visitStartTime - test_df_r.visitId).astype(np.int16)
        gc.collect()
        print(f'test_df_r: {test_df_r.shape}')

    with timer('save train data'):
        test_df_r.to_pickle('test_df_r', compression='gzip')
        del test_df_r
        gc.collect()

    with timer('reload train & test data'):
        train_df_r = pd.read_pickle('train_df_r', compression='gzip')
        test_df_r = pd.read_pickle('test_df_r', compression='gzip')
        print(f'train_df_r: {train_df_r.shape}, test_df_r: {test_df_r.shape}')

    return train_df_r, test_df_r


def get_window_data(train_df_r, test_df_r):
    def agg_most_common(df, _cols, n=3, window_type='u', gp_col=None):
        gp_cols = [id_col]
        if gp_col is not None:
            gp_cols.append(gp_col)

        df = df.loc[:, _cols + gp_cols].set_index(gp_cols)
        gc.collect()
        cnt = df.groupby(gp_cols)[_cols[0]].count()

        ids1 = cnt.loc[1 == cnt].index
        df1 = df.loc[ids1].copy()
        for _col in _cols:
            for k in range(1, n):
                df1[f'{_col}_{window_type}_{k}'] = -127
                df1[f'{_col}_{window_type}_{k}'] = df1[f'{_col}_{window_type}_{k}'].astype(df[_col].dtype)

        ids2 = cnt.loc[2 == cnt].index
        gp = df.loc[ids2].groupby(gp_cols)
        df2 = gp.max().join(gp.min(), rsuffix=f'_{window_type}_1')
        for _col in _cols:
            for k in range(2, n):
                df2[f'{_col}_{window_type}_{k}'] = -127
                df2[f'{_col}_{window_type}_{k}'] = df2[f'{_col}_{window_type}_{k}'].astype(df[_col].dtype)
            df2[f'{_col}_{window_type}_1'] = df2[f'{_col}_{window_type}_1'].astype(df[_col].dtype)

        def most_common(rows):
            _cnt = sorted(Counter(rows).items(), key=lambda pair: (pair[1], pair[0]), reverse=True)
            return [_cnt[i][0] if len(_cnt) > i else -127 for i in range(n)]

        ids3 = cnt.loc[cnt >= 3].index
        df3 = df.loc[ids3].groupby(gp_cols).agg(most_common)
        for _col in _cols:
            for k in range(1, n):
                df3[f'{_col}_{window_type}_{k}'] = df3[_col].str.get(k)
                df3[f'{_col}_{window_type}_{k}'] = df3[f'{_col}_{window_type}_{k}'].astype(df[_col].dtype)
            df3[_col] = df3[_col].str.get(0)
            df3[_col] = df3[_col].astype(df[_col].dtype)

        return df1.append([df2, df3], sort=False).rename(columns={_col: f'{_col}_{window_type}_0' for _col in _cols})

    with timer('most common 2'):
        cols = ['device.isMobile', 'totals.bounces', 'totals.newVisits', 'trafficSource.adwordsClickInfo.isVideoAd',
                'trafficSource.isTrueDirect', 'visit_weekend']
        train_df_w = agg_most_common(train_df_r, cols, n=2, window_type='w', gp_col='window_id')
        gc.collect()
        test_df_w = agg_most_common(test_df_r, cols, n=2, window_type='w', gp_col='window_id')
        gc.collect()
        print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

    with timer('most common 3'):
        cols = ['channelGrouping', 'customDimensions', 'device.browser', 'device.deviceCategory',
                'device.operatingSystem', 'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country',
                'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region', 'geoNetwork.subContinent',
                'trafficSource.adContent', 'trafficSource.adwordsClickInfo.adNetworkType',
                'trafficSource.adwordsClickInfo.slot', 'trafficSource.campaign', 'trafficSource.keyword',
                'trafficSource.medium', 'trafficSource.referralPath', 'trafficSource.source', 'ref_level_1',
                'ref_level_2', 'ref_level_3', 'ref_level_4', 'ref_level_n', 'src_level_1', 'src_level_2', 'src_level_3',
                'src_level_n', 'visit_year']
        train_df_w1 = agg_most_common(train_df_r, cols, n=3, window_type='w', gp_col='window_id')
        train_df_w = train_df_w.join(train_df_w1)
        del train_df_w1
        gc.collect()
        test_df_w1 = agg_most_common(test_df_r, cols, n=3, window_type='w', gp_col='window_id')
        test_df_w = test_df_w.join(test_df_w1)
        del test_df_w1
        gc.collect()
        print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

    with timer('most common 4'):
        cols = ['visit_quarter']
        train_df_w1 = agg_most_common(train_df_r, cols, n=4, window_type='w', gp_col='window_id')
        train_df_w = train_df_w.join(train_df_w1)
        del train_df_w1
        gc.collect()
        test_df_w1 = agg_most_common(test_df_r, cols, n=4, window_type='w', gp_col='window_id')
        test_df_w = test_df_w.join(test_df_w1)
        del test_df_w1
        gc.collect()
        print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

    with timer('most common 7'):
        cols = ['visit_month', 'visit_day', 'visit_hour', 'visit_week', 'visit_dayofweek', 'visit_dayofyear']
        train_df_w1 = agg_most_common(train_df_r, cols, n=7, window_type='w', gp_col='window_id')
        train_df_w = train_df_w.join(train_df_w1)
        del train_df_w1
        gc.collect()
        test_df_w1 = agg_most_common(test_df_r, cols, n=7, window_type='w', gp_col='window_id')
        test_df_w = test_df_w.join(test_df_w1)
        del test_df_w1
        gc.collect()
        print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

    def agg_numeric_cols(df, gdf, _cols, _agg_methods, window_type='u', gp_col=None, positive=False):
        gp_cols = [id_col]
        if gp_col is not None:
            gp_cols.append(gp_col)

        data_types = df.dtypes
        df = df.loc[:, _cols + gp_cols].copy()
        if positive:
            for _col in _cols:
                df.loc[df[_col] <= 0, _col] = np.nan

        gp = df.groupby(gp_cols)
        if 'mean' in _agg_methods:
            gdf = gdf.join(gp.mean().fillna(0))
            for _col in _cols:
                gdf[_col] = gdf[_col].astype(data_types[_col])
            gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_mean' for _col in _cols})
            gc.collect()
        if 'std' in _agg_methods:
            gdf = gdf.join(gp.std().fillna(0).astype(np.float32))
            gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_std' for _col in _cols})
            gc.collect()
        if 'sum' in _agg_methods:
            gdf = gdf.join(gp.sum().fillna(0))
            gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_sum' for _col in _cols})
            gc.collect()
        if 'min' in _agg_methods:
            gdf = gdf.join(gp.min().fillna(0))
            gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_min' for _col in _cols})
            for _col in _cols:
                gdf[f'{_col}_{window_type}_min'] = gdf[f'{_col}_{window_type}_min'].astype(data_types[_col])
            gc.collect()
        if 'max' in _agg_methods:
            gdf = gdf.join(gp.max().fillna(0))
            gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_max' for _col in _cols})
            for _col in _cols:
                gdf[f'{_col}_{window_type}_max'] = gdf[f'{_col}_{window_type}_max'].astype(data_types[_col])
            gc.collect()
        return gdf

    with timer('agg numeric cols'):
        cols = ['max_diff_time', 'max_price', 'min_diff_time', 'min_price', 'totals.extraRevenue', 'totals.hits',
                'totals.pageviews', 'totals.sessionQualityDim', 'totals.timeOnSite', 'totals.totalTransactionRevenue',
                'user_g_span', 'visit_day_pv', 'visit_hour_pv']
        agg_methods = ['mean', 'std', 'min', 'max']
        train_df_w = agg_numeric_cols(train_df_r, train_df_w, cols, agg_methods, window_type='w', gp_col='window_id',
                                      positive=True)
        gc.collect()
        test_df_w = agg_numeric_cols(test_df_r, test_df_w, cols, agg_methods, window_type='w', gp_col='window_id',
                                     positive=True)
        gc.collect()
        print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

        cols = ['visit_delay']
        agg_methods = ['mean', 'std', 'min', 'max']
        train_df_w = agg_numeric_cols(train_df_r, train_df_w, cols, agg_methods, window_type='w', gp_col='window_id',
                                      positive=False)
        gc.collect()
        test_df_w = agg_numeric_cols(test_df_r, test_df_w, cols, agg_methods, window_type='w', gp_col='window_id',
                                     positive=False)
        gc.collect()
        print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

        cols = ['mean_diff_time', 'mean_price', 'mean_time', 'std_diff_time', 'std_price', 'max_price_ratio',
                'mean_price_ratio', 'min_price_ratio', 'totals.transactions', 'trafficSource.adwordsClickInfo.page',
                'user_v_pv', 'user_s_pv', 'user_s_idle', 'user_s_span', 'user_v_idle', 'user_v_span']
        agg_methods = ['mean', 'min', 'max']
        train_df_w = agg_numeric_cols(train_df_r, train_df_w, cols, agg_methods, window_type='w', gp_col='window_id',
                                      positive=True)
        gc.collect()
        test_df_w = agg_numeric_cols(test_df_r, test_df_w, cols, agg_methods, window_type='w', gp_col='window_id',
                                     positive=True)
        gc.collect()
        print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

        cols = ['ref_level_cnt', 'src_level_cnt', 'visit_idle']
        agg_methods = ['mean', 'min', 'max']
        train_df_w = agg_numeric_cols(train_df_r, train_df_w, cols, agg_methods, window_type='w', gp_col='window_id',
                                      positive=False)
        gc.collect()
        test_df_w = agg_numeric_cols(test_df_r, test_df_w, cols, agg_methods, window_type='w', gp_col='window_id',
                                     positive=False)
        gc.collect()
        print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

        cols = ['user_w_pv', 'user_w_idle', 'user_w_span']
        agg_methods = ['max']
        train_df_w = agg_numeric_cols(train_df_r, train_df_w, cols, agg_methods, window_type='w', gp_col='window_id',
                                      positive=False)
        gc.collect()
        test_df_w = agg_numeric_cols(test_df_r, test_df_w, cols, agg_methods, window_type='w', gp_col='window_id',
                                     positive=False)
        gc.collect()
        print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

        cols = ['visitStartTime']
        agg_methods = ['min', 'max']
        train_df_w = agg_numeric_cols(train_df_r, train_df_w, cols, agg_methods, window_type='w', gp_col='window_id',
                                      positive=False)
        gc.collect()
        test_df_w = agg_numeric_cols(test_df_r, test_df_w, cols, agg_methods, window_type='w', gp_col='window_id',
                                     positive=False)
        gc.collect()
        print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

        cols = ['target']
        agg_methods = ['sum']
        train_df_w = agg_numeric_cols(train_df_r, train_df_w, cols, agg_methods, window_type='w', gp_col='window_id',
                                      positive=False)
        gc.collect()
        test_df_w = agg_numeric_cols(test_df_r, test_df_w, cols, agg_methods, window_type='w', gp_col='window_id',
                                     positive=False)
        gc.collect()
        print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

    def stat_window_cnt(df, gdf, _cols=None, window_type='u', gp_col=None):
        _cols = _cols if _cols else ['visitId', 'session_id']
        gp_cols = [id_col]
        if gp_col is not None:
            gp_cols.append(gp_col)

        df = df[_cols + gp_cols]
        gc.collect()
        gdf = gdf.join(
            df.groupby(gp_cols)[_cols].nunique().rename(columns={_col: f'{_col}_{window_type}_cnt' for _col in _cols}))
        gc.collect()
        gdf[f'visitId_{window_type}_cnt'] = gdf[f'visitId_{window_type}_cnt'].astype(np.int16)
        gdf[f'session_id_{window_type}_cnt'] = gdf[f'session_id_{window_type}_cnt'].astype(np.int8)
        gc.collect()
        return gdf

    with timer('stat window cnt'):
        train_df_w = stat_window_cnt(train_df_r, train_df_w, window_type='w', gp_col='window_id')
        gc.collect()
        test_df_w = stat_window_cnt(test_df_r, test_df_w, window_type='w', gp_col='window_id')
        gc.collect()
        print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

    def concat_by_window(df, gdf, _cols, window_type='u', gp_col=None):
        gp_cols = [id_col]
        if gp_col is not None:
            gp_cols.append(gp_col)

        df = df.loc[:, _cols + gp_cols].set_index(gp_cols)
        gc.collect()
        cnt = df.groupby(gp_cols)[_cols[0]].count()

        ids1 = cnt.loc[1 == cnt].index
        df1 = df.loc[ids1]

        ids2 = cnt.loc[cnt >= 2].index
        df2 = df.loc[ids2].groupby(gp_cols).agg(lambda rows: ' '.join(rows))
        gdf = gdf.join(df1.append(df2, sort=False).rename(columns={_col: f'{window_type}_{_col}' for _col in _cols}))

        return gdf

    with timer('concat texts'):
        cols = ['exitScreenNames_src', 'promoNames_src', 'productNames_src', 'productCategorys_src',
                'hitProductNames_src', 'device.browser_src', 'trafficSource.adContent_src', 'trafficSource.keyword_src',
                'trafficSource.referralPath_src', 'trafficSource.source_src']
        train_df_w = concat_by_window(train_df_r, train_df_w, cols, window_type='w', gp_col='window_id')
        gc.collect()
        test_df_w = concat_by_window(test_df_r, test_df_w, cols, window_type='w', gp_col='window_id')
        gc.collect()
        print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

    with timer('save data'):
        train_df_w.to_pickle('train_df_w', compression='gzip')
        del train_df_w
        gc.collect()
        test_df_w.to_pickle('test_df_w', compression='gzip')
        del test_df_w
        gc.collect()

    with timer('reload train & test data'):
        train_df_w = pd.read_pickle('train_df_w', compression='gzip')
        test_df_w = pd.read_pickle('test_df_w', compression='gzip')
        print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

    return train_df_w, test_df_w


def get_user_data(train_df_r, test_df_r, train_df_w, test_df_w):
    def expand_most_common(_df, _cols, n=3, window_type='u', gp_col='window_id', sort_col='visitStartTime'):
        gp_cols = [id_col]
        if gp_col is not None:
            gp_cols.append(gp_col)

        _df = _df.loc[:, _cols + gp_cols + [sort_col]].sort_values(by=sort_col)
        gc.collect()
        cnt = _df[id_col].value_counts()

        ids1 = cnt.loc[1 == cnt].index
        df1 = _df.loc[_df[id_col].isin(ids1)].copy()
        for _col in _cols:
            for k in range(1, n):
                df1[f'{_col}_{window_type}_{k}'] = -127
                df1[f'{_col}_{window_type}_{k}'] = df1[f'{_col}_{window_type}_{k}'].astype(_df[_col].dtype)

        def most_common(eles):
            _cnt = Counter()
            commons = []
            for ele in eles:
                _cnt.update([ele])
                cnt_pair = sorted(_cnt.items(), key=lambda pair: (pair[1], pair[0]), reverse=True)
                commons.append([cnt_pair[i][0] if len(cnt_pair) > i else -127 for i in range(n)])
            return commons

        ids2 = cnt.loc[cnt >= 2].index
        df2 = _df.loc[_df[id_col].isin(ids2)]
        gdf = df2.groupby(id_col)[_cols].transform(most_common)
        gdf = gdf.loc[df2.groupby(gp_cols)[sort_col].idxmax()]
        for _col in _cols:
            for k in range(1, n):
                gdf[f'{_col}_{window_type}_{k}'] = gdf[_col].str.get(k)
                gdf[f'{_col}_{window_type}_{k}'] = gdf[f'{_col}_{window_type}_{k}'].astype(_df[_col].dtype)
            gdf[_col] = gdf[_col].str.get(0)
            gdf[_col] = gdf[_col].astype(_df[_col].dtype)
        for _col in gp_cols:
            gdf[_col] = df2[_col]

        gdf = df1.drop(sort_col, axis=1).append(gdf, sort=False).rename(
            columns={_col: f'{_col}_{window_type}_0' for _col in _cols}).set_index(gp_cols)
        gc.collect()

        return gdf

    with timer('most common 2'):
        cols = ['device.isMobile', 'totals.bounces', 'totals.newVisits', 'trafficSource.adwordsClickInfo.isVideoAd',
                'trafficSource.isTrueDirect', 'visit_weekend']
        train_df_u = expand_most_common(train_df_r, cols, n=2, window_type='u')
        gc.collect()
        test_df_u = expand_most_common(test_df_r, cols, n=2, window_type='u')
        gc.collect()
        print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

    with timer('most common 3'):
        cols = ['channelGrouping', 'customDimensions', 'device.browser', 'device.deviceCategory',
                'device.operatingSystem', 'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country',
                'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region', 'geoNetwork.subContinent',
                'trafficSource.adContent', 'trafficSource.adwordsClickInfo.adNetworkType',
                'trafficSource.adwordsClickInfo.slot', 'trafficSource.campaign', 'trafficSource.keyword',
                'trafficSource.medium', 'trafficSource.referralPath', 'trafficSource.source', 'ref_level_1',
                'ref_level_2', 'ref_level_3', 'ref_level_4', 'ref_level_n', 'src_level_1', 'src_level_2', 'src_level_3',
                'src_level_n', 'visit_year']
        train_df_u1 = expand_most_common(train_df_r, cols, n=3, window_type='u')
        train_df_u = train_df_u.join(train_df_u1)
        del train_df_u1
        gc.collect()
        test_df_u1 = expand_most_common(test_df_r, cols, n=3, window_type='u')
        test_df_u = test_df_u.join(test_df_u1)
        del test_df_u1
        gc.collect()
        print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

    with timer('most common 4'):
        cols = ['visit_quarter']
        train_df_u1 = expand_most_common(train_df_r, cols, n=4, window_type='u')
        train_df_u = train_df_u.join(train_df_u1)
        del train_df_u1
        gc.collect()
        test_df_u1 = expand_most_common(test_df_r, cols, n=4, window_type='u')
        test_df_u = test_df_u.join(test_df_u1)
        del test_df_u1
        gc.collect()
        print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

    with timer('most common 7'):
        cols = ['visit_month', 'visit_day', 'visit_hour', 'visit_week', 'visit_dayofweek', 'visit_dayofyear']
        train_df_u1 = expand_most_common(train_df_r, cols, n=7, window_type='u')
        train_df_u = train_df_u.join(train_df_u1)
        del train_df_u1
        gc.collect()
        test_df_u1 = expand_most_common(test_df_r, cols, n=7, window_type='u')
        test_df_u = test_df_u.join(test_df_u1)
        del test_df_u1
        gc.collect()
        print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

    def expand_numeric_cols(_df, _gdf, _cols, _expand_methods, window_type='u', gp_col='window_id', positive=False,
                            sort_col='visitStartTime'):
        gp_cols = [id_col]
        if gp_col is not None:
            gp_cols.append(gp_col)

        data_types = _df.dtypes
        cnt = _df[id_col].value_counts()

        ids1 = cnt.loc[1 == cnt].index
        df1 = _df.loc[_df[id_col].isin(ids1)]
        gdf1 = pd.DataFrame()
        for _col in gp_cols:
            gdf1[_col] = df1[_col]
        for _col in _cols:
            column = df1[_col]
            for agg_method in _expand_methods:
                if 'std' != agg_method:
                    gdf1[f'{_col}_{window_type}_{agg_method}'] = column
                else:
                    gdf1[f'{_col}_{window_type}_std'] = 0
                    gdf1[f'{_col}_{window_type}_std'] = gdf1[f'{_col}_{window_type}_std'].astype(np.float32)
        gdf1 = gdf1.set_index(gp_cols)
        del df1
        gc.collect()

        ids2 = cnt.loc[cnt > 1].index
        _df = _df.loc[_df[id_col].isin(ids2)]
        _df = _df.loc[:, _cols + gp_cols + ([] if sort_col in _cols else [sort_col])].sort_values(by=sort_col).copy()
        if positive:
            for _col in _cols:
                _df.loc[_df[_col] <= 0, _col] = np.nan
        gc.collect()

        gp = _df.groupby(id_col)[_cols].expanding()
        gdf = None
        if 'mean' in _expand_methods:
            gdf = gp.mean().fillna(0)
            for _col in _cols:
                gdf[_col] = gdf[_col].astype(data_types[_col])
            gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_mean' for _col in _cols})
            gc.collect()
        if 'std' in _expand_methods:
            cur_gdf = gp.std().fillna(0).astype(np.float32)
            gdf = gdf.join(cur_gdf) if gdf is not None else cur_gdf
            gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_std' for _col in _cols})
            gc.collect()
        if 'sum' in _expand_methods:
            cur_gdf = gp.sum().fillna(0)
            gdf = gdf.join(cur_gdf) if gdf is not None else cur_gdf
            gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_sum' for _col in _cols})
            gc.collect()
        if 'min' in _expand_methods:
            cur_gdf = gp.min().fillna(0)
            gdf = gdf.join(cur_gdf) if gdf is not None else cur_gdf
            gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_min' for _col in _cols})
            for _col in _cols:
                gdf[f'{_col}_{window_type}_min'] = gdf[f'{_col}_{window_type}_min'].astype(data_types[_col])
            gc.collect()
        if 'max' in _expand_methods:
            cur_gdf = gp.max().fillna(0)
            gdf = gdf.join(cur_gdf) if gdf is not None else cur_gdf
            gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_max' for _col in _cols})
            for _col in _cols:
                gdf[f'{_col}_{window_type}_max'] = gdf[f'{_col}_{window_type}_max'].astype(data_types[_col])
            gc.collect()

        gdf.index = gdf.index.droplevel()
        gdf = gdf.loc[_df.groupby(gp_cols)[sort_col].idxmax()]
        for _col in gp_cols:
            gdf[_col] = _df[_col]

        gdf = gdf1.append(gdf.set_index(gp_cols), sort=False)
        _gdf = _gdf.join(gdf)
        del _df, gdf1, gdf, gp
        gc.collect()

        return _gdf

    with timer('expand numeric cols'):
        cols = ['max_diff_time', 'max_price', 'min_diff_time', 'min_price', 'totals.extraRevenue', 'totals.hits',
                'totals.pageviews', 'totals.sessionQualityDim', 'totals.timeOnSite', 'totals.totalTransactionRevenue',
                'user_g_span', 'visit_day_pv', 'visit_hour_pv']
        expand_methods = ['mean', 'std', 'min', 'max']
        train_df_u = expand_numeric_cols(train_df_r, train_df_u, cols, expand_methods, positive=True)
        gc.collect()
        test_df_u = expand_numeric_cols(test_df_r, test_df_u, cols, expand_methods, positive=True)
        gc.collect()
        print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

        cols = ['visit_delay']
        expand_methods = ['mean', 'std', 'min', 'max']
        train_df_u = expand_numeric_cols(train_df_r, train_df_u, cols, expand_methods, positive=False)
        gc.collect()
        test_df_u = expand_numeric_cols(test_df_r, test_df_u, cols, expand_methods, positive=False)
        gc.collect()
        print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

        cols = ['mean_diff_time', 'mean_price', 'mean_time', 'std_diff_time', 'std_price', 'max_price_ratio',
                'mean_price_ratio', 'min_price_ratio', 'totals.transactions', 'trafficSource.adwordsClickInfo.page',
                'user_v_pv', 'user_s_pv', 'user_s_idle', 'user_s_span', 'user_v_idle', 'user_v_span']
        expand_methods = ['mean', 'min', 'max']
        train_df_u = expand_numeric_cols(train_df_r, train_df_u, cols, expand_methods, positive=True)
        gc.collect()
        test_df_u = expand_numeric_cols(test_df_r, test_df_u, cols, expand_methods, positive=True)
        gc.collect()
        print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

        cols = ['ref_level_cnt', 'src_level_cnt', 'visit_idle']
        expand_methods = ['mean', 'min', 'max']
        train_df_u = expand_numeric_cols(train_df_r, train_df_u, cols, expand_methods, positive=False)
        gc.collect()
        test_df_u = expand_numeric_cols(test_df_r, test_df_u, cols, expand_methods, positive=False)
        gc.collect()
        print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

        cols = ['visitStartTime']
        expand_methods = ['min', 'max']
        train_df_u = expand_numeric_cols(train_df_r, train_df_u, cols, expand_methods, positive=False)
        gc.collect()
        test_df_u = expand_numeric_cols(test_df_r, test_df_u, cols, expand_methods, positive=False)
        gc.collect()
        train_df_u['user_u_span'] = train_df_u['visitStartTime_u_max'] - train_df_u['visitStartTime_u_min']
        test_df_u['user_u_span'] = test_df_u['visitStartTime_u_max'] - test_df_u['visitStartTime_u_min']
        print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

    def expand_target_cols(_df, _gdf, _expand_methods, t_col='target_w_sum', window_type='u',
                           sort_col='visitStartTime_w_min', positive=False):
        cnt = _df.groupby(id_col)[sort_col].count()

        ids1 = cnt.loc[1 == cnt].index
        df1 = _df.loc[ids1, [t_col]].copy()
        column = df1[t_col]
        for agg_method in _expand_methods:
            if 'std' != agg_method:
                df1[f'target_{window_type}_{agg_method}'] = column
            else:
                df1[f'target_{window_type}_std'] = 0
                df1[f'target_{window_type}_std'] = df1[f'target_{window_type}_std'].astype(np.int64)
        df1 = df1.drop(t_col, axis=1)
        gc.collect()

        ids2 = cnt.loc[cnt > 1].index
        _df = _df.loc[ids2, [t_col, sort_col]].sort_values(by=sort_col).copy()
        if positive:
            _df.loc[_df[t_col] <= 0, t_col] = np.nan
        gc.collect()

        gp = _df.groupby(id_col)[[t_col]].expanding()
        gdf = None
        for agg_method in _expand_methods:
            cur_gdf = gp.agg(agg_method).fillna(0).astype(np.int64).rename(
                columns={t_col: f'target_{window_type}_{agg_method}'})
            gdf = gdf.join(cur_gdf) if gdf is not None else cur_gdf
            del cur_gdf
            gc.collect()

        gdf.index = gdf.index.droplevel()
        gdf = df1.append(gdf, sort=False)
        _gdf = _gdf.join(gdf)
        del df1, _df, gdf, gp
        gc.collect()

        return _gdf

    with timer('expand target'):
        expand_methods = ['mean', 'std', 'sum', 'min', 'max']
        train_df_u = expand_target_cols(train_df_w, train_df_u, expand_methods, positive=True)
        gc.collect()
        test_df_u = expand_target_cols(test_df_w, test_df_u, expand_methods, positive=True)
        gc.collect()
        print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

    def expand_window_cnt(_df, gdf, _cols=None, window_type='u', sort_col='visitStartTime_w_min'):
        _cols = _cols if _cols else ['visitId_w_cnt', 'session_id_w_cnt']

        _df = _df.sort_values(by=sort_col)
        gp = _df.groupby(id_col)

        cur_df = gp[sort_col].cumcount().astype(np.uint8)
        cur_df.name = f'window_id_{window_type}_cnt'
        gdf = gdf.join(cur_df)
        gc.collect()

        gdf = gdf.join(gp[_cols].cumsum().rename(
            columns={_col: _col.replace('_w_', f'_{window_type}_') for _col in _cols}))
        del _df, gp, cur_df
        gc.collect()

        return gdf

    with timer('expand window cnt'):
        train_df_u = expand_window_cnt(train_df_w, train_df_u)
        gc.collect()
        test_df_u = expand_window_cnt(test_df_w, test_df_u)
        gc.collect()
        print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

    def expanded_concat_by_window(_df, _gdf, _cols, window_type='u', sort_col='visitStartTime_w_min'):
        cnt = _df.groupby(id_col)[sort_col].count()

        ids1 = cnt.loc[1 == cnt].index
        df1 = _df.loc[ids1, _cols].rename(columns={_col: _col.replace('w_', f'{window_type}_') for _col in _cols})
        gc.collect()

        def expanded_concat(eles):
            infos = []
            info = ''
            for ele in eles:
                info += ele + ' '
                infos.append(info)
            return infos

        ids2 = cnt.loc[cnt > 1].index
        _df = _df.loc[ids2].sort_values(by=sort_col)
        gc.collect()
        gdf = _df.groupby(id_col)[_cols].transform(expanded_concat).rename(
            columns={_col: _col.replace('w_', f'{window_type}_') for _col in _cols})
        gc.collect()
        gdf = df1.append(gdf, sort=False)
        _gdf = _gdf.join(gdf) if _gdf is not None else gdf
        del _df, df1, gdf
        gc.collect()

        return _gdf

    with timer('expand texts'):
        cols = ['w_productNames_src', 'w_productCategorys_src', 'w_hitProductNames_src']
        train_df_u = expanded_concat_by_window(train_df_w, train_df_u, cols)
        gc.collect()
        test_df_u = expanded_concat_by_window(test_df_w, test_df_u, cols)
        gc.collect()
        print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

    with timer('save data'):
        train_df_u.to_pickle('train_df_u', compression='gzip')
        del train_df_u
        gc.collect()
        test_df_u.to_pickle('test_df_u', compression='gzip')
        del test_df_u
        gc.collect()

    with timer('reload train & test data'):
        train_df_u = pd.read_pickle('train_df_u', compression='gzip')
        test_df_u = pd.read_pickle('test_df_u', compression='gzip')
        print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

    return train_df_u, test_df_u


def process_data(train_df, test_df, sub_ids):
    print(f'train_df isnull: {np.sum(train_df.isnull().values)}, test_df isnull: {np.sum(test_df.isnull().values)}')

    with timer('sort'):
        sort_col = 'visitStartTime_w_min'
        train_df = train_df.reset_index()
        gc.collect()
        train_df = train_df.sort_values(by=[id_col, sort_col])
        gc.collect()
        train_df = train_df.reset_index(drop=True)
        gc.collect()

        test_df = test_df.reset_index()
        gc.collect()
        test_df = test_df.sort_values(by=[id_col, sort_col])
        gc.collect()
        test_df = test_df.reset_index(drop=True)
        gc.collect()

    with timer('get target info'):
        def label_target(tfs, tfe, vtfs, vtfe):
            tr_df = train_df.loc[(((train_df[tm_col1] >= tfs) & (train_df[tm_col1] < tfe))
                                  | ((train_df[tm_col2] >= tfs) & (train_df[tm_col2] < tfe)))]
            vids = train_df.loc[(((train_df[tm_col1] >= vtfs) & (train_df[tm_col1] < vtfe))
                                 | ((train_df[tm_col2] >= vtfs) & (train_df[tm_col2] < vtfe))), id_col].unique()
            tr_df = tr_df.loc[~tr_df[id_col].isin(vids)]
            gc.collect()
            tr_df = tr_df.loc[tr_df.groupby(id_col)[tm_col1].idxmax()].copy()
            tr_df['target'] = 0

            return tr_df

        tfs1 = pd.to_datetime('2017-05-01').timestamp()
        tfe1 = pd.to_datetime('2017-10-01').timestamp()
        vtfs1 = pd.to_datetime('2017-12-01').timestamp()
        vtfe1 = pd.to_datetime('2018-02-01').timestamp()

        tfs2 = pd.to_datetime('2018-01-01').timestamp()
        tfe2 = pd.to_datetime('2018-06-01').timestamp()
        vtfs2 = pd.to_datetime('2018-08-01').timestamp()
        vtfe2 = pd.to_datetime('2018-10-01').timestamp()

        tm_col1 = 'visitStartTime_w_min'
        tm_col2 = 'visitStartTime_w_max'
        tr_df1 = label_target(tfs1, tfe1, vtfs1, vtfe1)
        gc.collect()
        tr_df2 = label_target(tfs2, tfe2, vtfs2, vtfe2)
        gc.collect()
        print(f'tr_df1: {tr_df1.shape}, tr_df2: {tr_df2.shape}')

        ids3 = train_df.loc[(((train_df[tm_col1] >= tfe1) & (train_df[tm_col2] < tfs2))
                             | (train_df[tm_col2] < tfs1)), id_col].unique()
        tr_df3 = train_df.loc[train_df[id_col].isin(ids3)]
        tr_df3 = tr_df3.loc[tr_df3.groupby(id_col)[tm_col1].idxmax()]
        tr_df3 = tr_df3.loc[(((tr_df3[tm_col1] >= tfe1) & (tr_df3[tm_col2] < tfs2)) | (tr_df3[tm_col2] < tfs1))].copy()
        gc.collect()
        tr_df3['target'] = 0
        print(f'tr_df3: {tr_df3.shape}')

        cnt = train_df[id_col].value_counts()
        ids4 = cnt.loc[cnt > 1].index.values
        tr_df4 = train_df.loc[train_df[id_col].isin(ids4)].copy()
        gc.collect()
        print(f'tr_df4: {tr_df4.shape}')

        t_col = 'target_w_sum'
        tr_df4['target'] = -1
        gp = tr_df4.groupby(id_col)
        cols = []
        for i in range(2, 5):
            col = f'{t_col}_{i}'
            cols.append(col)
            tr_df4[col] = gp[t_col].shift(-i)
            col = f'{tm_col1}_{i}'
            cols.append(col)
            tr_df4[col] = gp[tm_col1].shift(-i)
            col = f'{tm_col2}_{i}'
            cols.append(col)
            tr_df4[col] = gp[tm_col2].shift(-i)
        for i in range(2, 5):
            ind4 = (tr_df4['target'] < 0) & tr_df4[f'{t_col}_{i}'].notnull() & (
                tr_df4[f'{tm_col2}_{i}'] - tr_df4[tm_col1] > 60 * 24 * 3600)
            ind40 = ind4 & (tr_df4[f'{tm_col1}_{i}'] - tr_df4[tm_col2] > 9 * 30 * 24 * 3600)
            tr_df4.loc[ind40, 'target'] = 0
            ind41 = ind4 & (tr_df4[f'{tm_col1}_{i}'] - tr_df4[tm_col2] <= 9 * 30 * 24 * 3600)
            tr_df4.loc[ind41, 'target'] = tr_df4.loc[ind41, f'{t_col}_{i}']
        tr_df4 = tr_df4.loc[tr_df4.target >= 0].drop(cols, axis=1)
        gc.collect()
        print(f'tr_df4: {tr_df4.shape}')

        train_df = tr_df1.append([tr_df2, tr_df3, tr_df4], sort=False)
        del tr_df1, tr_df2, tr_df3, tr_df4
        gc.collect()
        train_df = train_df.sort_values(by=[id_col, sort_col])
        gc.collect()
        train_df = train_df.reset_index(drop=True)
        gc.collect()

        train_df = train_df.drop('target_w_sum', axis=1)
        gc.collect()
        test_df = test_df.drop('target_w_sum', axis=1)
        gc.collect()
        print(f'train_df: {train_df.shape}, test_df: {test_df.shape}')

    with timer('remove non train data'):
        train_df = train_df.drop('window_id', axis=1)
        gc.collect()
        test_df = test_df.loc[0 == test_df.window_id].drop('window_id', axis=1)
        gc.collect()
        print(f'train_df: {train_df.shape}, test_df: {test_df.shape}')

    with timer('check and sort with sub_ids'):
        s = test_df[id_col].unique()
        print(f'ids@test_df: {s.shape}, sub_ids: {sub_ids.shape}')
        c_sub_ids = np.intersect1d(s, sub_ids)
        print(f'same sub_ids: {c_sub_ids.shape}')

        test_df = test_df.set_index(id_col)
        gc.collect()
        test_df = sub_ids.to_frame().join(test_df, on=id_col)
        gc.collect()
        print(f'test_df isnull: {np.sum(test_df.isnull().values)}')

    with timer('remove still cols'):
        def find_still_cols(df, diff_num_threshold=30):
            _still_cols1, _still_cols2 = [], []
            for _col in df.columns:
                _cnt = df[_col].value_counts(dropna=False)
                if _cnt.shape[0] <= 1:
                    _still_cols1.append(_col)
                elif df.shape[0] - _cnt.iloc[0] < diff_num_threshold:
                    _still_cols2.append(_col)
            return _still_cols1, _still_cols2

        still_cols1, still_cols2 = find_still_cols(train_df)
        print(f'still_cols1({len(still_cols1)}): {still_cols1}')
        print(f'still_cols2({len(still_cols2)}): {still_cols2}')
        still_cols = still_cols1 + still_cols2
        train_df = train_df.drop(still_cols, axis=1)
        gc.collect()
        test_df = test_df.drop(still_cols, axis=1)
        gc.collect()
        print(f'train_df: {train_df.shape}, test_df: {test_df.shape}')

    with timer('remove same cols'):
        def find_same_cols(df, diff_num_threshold=30, detail=False):
            _same_cols = set()
            for col1 in df.columns:
                if col1 not in _same_cols:
                    column1 = df[col1]
                    for col2 in df.columns:
                        if col2 > col1 and col2 not in _same_cols:
                            diff_num = np.sum(column1 != df[col2])
                            if diff_num < diff_num_threshold:
                                if detail:
                                    print(f'{col1} - {col2} = {diff_num}')
                                _same_cols.add(col2)
            return list(_same_cols)

        same_cols = sorted(find_same_cols(train_df, detail=True))
        print(f'same_cols({len(same_cols)}): {same_cols}')
        train_df = train_df.drop(same_cols, axis=1)
        gc.collect()
        test_df = test_df.drop(same_cols, axis=1)
        gc.collect()
        print(f'train_df: {train_df.shape}, test_df: {test_df.shape}')

    with timer('replace never seen values in test_df'):
        cols = [col for col in test_df.columns if col[-1].isdigit()]
        for col in cols:
            cnt = Counter(train_df[col])
            cnt_pair = sorted(cnt.items(), key=lambda pair: (pair[1], pair[0]), reverse=True)
            unseen_value = -127 if -127 in cnt else '-127' if '-127' in cnt else cnt_pair[0][0]
            data_type = test_df[col].dtype
            test_df[col] = test_df[col].apply(lambda ele: unseen_value if ele not in cnt else ele).astype(data_type)
            del cnt, cnt_pair
            gc.collect()

    with timer('vectorize text cols'):
        text_cols = [col for col in test_df.columns if col.endswith('_src')]
        print('-----------------------------cols----------------------------')
        print(f'text_cols({len(text_cols)}): {text_cols}')
        non_text_cols = [col for col in test_df.columns if not col.endswith('_src')]
        print(f'non_text_cols({len(non_text_cols)}): {non_text_cols}')
        print('-------------------------------------------------------------')

        tr_x_texts = []
        ts_x_texts = []
        for col in text_cols:
            if 'keyword_src' in col:
                tvr = TfidfVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', strip_accents='unicode', min_df=2)
            elif 'referralPath_src' in col:
                tvr = TfidfVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', strip_accents='unicode', min_df=5)
            else:
                tvr = TfidfVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', strip_accents='unicode')
            tr_x_text = tvr.fit_transform(train_df[col])
            tr_x_texts.append(tr_x_text)
            ts_x_text = tvr.transform(test_df[col])
            ts_x_texts.append(ts_x_text)
            print(f'train {col}: {tr_x_text.shape}, test {col}: {ts_x_text.shape}')

            print(f'-----------------------------{col}----------------------------')
            print(list(tvr.get_feature_names()))
            print(f'--------------------------------------------------------------')

            del tvr, tr_x_text, ts_x_text
            gc.collect()

        tr_x_text = hstack(tr_x_texts, dtype=np.float32).tocsr()
        del tr_x_texts
        gc.collect()
        ts_x_text = hstack(ts_x_texts, dtype=np.float32).tocsr()
        del ts_x_texts
        gc.collect()
        print(f'tr_x_text: {tr_x_text.shape}, ts_x_text: {ts_x_text.shape}')

        train_df = train_df.drop(text_cols, axis=1)
        gc.collect()
        test_df = test_df.drop(text_cols, axis=1)
        gc.collect()
        print(f'train_df: {train_df.shape}, test_df: {test_df.shape}')

    return train_df, test_df, tr_x_text, ts_x_text


def run():
    sys.stderr = sys.stdout = open(os.path.join('get_src_data_log'), 'w')

    tr_ids, sub_ids = get_ids()
    train_df_r, test_df_r = get_record_data(tr_ids, sub_ids)
    train_df_w, test_df_w = get_window_data(train_df_r, test_df_r)
    train_df_u, test_df_u = get_user_data(train_df_r, test_df_r, train_df_w, test_df_w)

    with timer('join data'):
        del train_df_r, test_df_r
        gc.collect()

        train_df = train_df_w.join(train_df_u)
        test_df = test_df_w.join(test_df_u)
        print(f'train_df: {train_df.shape}, test_df: {test_df.shape}')

        del train_df_w, test_df_w, train_df_u, test_df_u
        gc.collect()

    train_df, test_df, tr_x_text, ts_x_text = process_data(train_df, test_df, sub_ids)

    with timer('save data'):
        train_df.to_pickle('train_df', compression='gzip')
        del train_df
        gc.collect()
        test_df.to_pickle('test_df', compression='gzip')
        del test_df
        gc.collect()

        joblib.dump(tr_x_text, 'tr_x_text', compress=('gzip', 3))
        del tr_x_text
        gc.collect()
        joblib.dump(ts_x_text, 'ts_x_text', compress=('gzip', 3))
        del ts_x_text
        gc.collect()
 

if __name__ == '__main__':
    run()

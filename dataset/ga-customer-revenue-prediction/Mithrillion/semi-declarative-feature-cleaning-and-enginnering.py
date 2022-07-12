"""
This script demonstrates how to write reuseable feature engineering code that can be easily modified to
test different feature selection, combination and interactions as you receive more feedback from model
training. It is designed in a way that only the declarative part of the script needs to be altered for
a new sklearn-ready dataset to be generated. It uses a number of flexible helper functions to ensure that
as little code as possible needs to be changed in the process.

The script currently depends on my processed data files from another kernel, but the idea should work for
any lightly preprocessed dataset, and this script should be able to be easily adapted to any datasets 
that use Pandas' category type for categorical variables. I chose the category type here mainly for 
keeping consistent encoding between train and test, as well as the ease of interfacing with LGBM.
"""

from os import path
from itertools import product
import re

import numpy as np
import pandas as pd

ROOT = '../input/geonetwork-feature-conflict-resolution/'
OUTPUT = '.'

TRAIN_FNAME = 'manual_geo_fix_train.pkl'
TEST_FNANE = 'manual_geo_fix_test.pkl'

##########################################################################
# set preprocessing parameters and constants
# set cutoff criteria for categorical attributes. Categories in selected columns with
# occurrences below COUNT_CUTOFF or positive class (revenue > 0) occurrences below
# POS_CLASS_CUTOFF will be converted to NaN
COUNT_CUTOFF = 10
POS_CLASS_CUTOFF = 0

GEO_HIERARCHY = [
    'geoNetwork.city', 'geoNetwork.region', 'geoNetwork.country',
    'geoNetwork.subContinent', 'geoNetwork.continent'
]

# define columns to drop
COLS_TO_DROP = [
    'trafficSource.adContent', 'trafficSource.medium',
    'trafficSource.adwordsClickInfo.gclId', 'trafficSource.keyword',
    'trafficSource.referralPath', 'date', 'sessionId', 'visitId',
    'visitStartTime', 'totals.bounces', 'totals.newVisits'
]  # totals.bounces and totals.newVisits are either 1 or NA, so convert to boolean below

# define columns whose missingness should be added as new features
MISSINGNESS_COLUMNS_TO_ADD = [
    'totals.bounces', 'totals.newVisits', 'trafficSource.adContent',
    'trafficSource.keyword', 'trafficSource.referralPath'
]

# time attributes to extract from visitStartTime
TIMEGROUP_EXTRACTION_LIST = ['year', 'month', 'dayofweek']

# geographical features to aggregate by (together with time)
GEO_LEVEL_AGG_LIST = ['geoNetwork.city', 'geoNetwork.country']
# possible aggregate functions to use
AGG_FUNC_DICT = {'mean': np.mean, 'max': np.max, 'total': np.sum}
# time features to aggregate by (together with geographical features)
GEO_TEMPERAL_TIME_GROUPS = ['year', 'month', 'dayofweek']
# attributes to aggregate, along with aggregate functions to use from AGG_FUNC_DICT
GEO_TEMPERAL_ATTR_DICT = {
    'totals.hits': ['mean', 'total'],
    'totals.pageviews': ['mean', 'total'],
    'hit_rate': ['mean']
}
USER_AGG_ATTR_DICT = {
    'totals.hits': ['mean', 'total'],
    'totals.pageviews': ['mean', 'total'],
    'miss_totals.bounces': ['mean', 'total'],
    'hit_rate': ['mean']
}
DOMAIN_AGG_ATTR_DICT = {
    'totals.hits': ['mean'],
    'totals.pageviews': ['mean'],
    'miss_totals.bounces': ['mean'],
    'hit_rate': ['mean']
}

#########################################################################
# define auxiliary functions

# a context manager to safely manipulate categorical-valued columns
class Decategorise:
    """
    A context manager class to deal with cases where pandas, sklearn etc. cannot deal with categorical types.
    By using 'with Decategorise(df) as decat_df:', when within the context, the categorical columns of the
    dataframe are converted to simple types (str, int, float etc.) or encodings (integer codes). When exiting
    the context, the columns are reverted to their original categories. Extra cares needs to be taken when
    adding new unique values to the columns, as this is not handled automatically!
    """
    def __init__(self, data_frame: pd.DataFrame, as_code=False):
        self.data_frame = data_frame
        self.as_code = as_code
        self.categorical_cols = [
            c for c in self.data_frame.columns
            if str(self.data_frame[c].dtype) == 'category'
        ]
        self.categories_dict = {
            c: self.data_frame[c].cat.categories
            for c in self.categorical_cols
        }

    def __enter__(self):
        if self.as_code:
            for col in self.categorical_cols:
                self.data_frame[col] = self.data_frame[col].cat.codes
        else:
            for col in self.categorical_cols:
                # potential bug: if type is Object but not string, might cause groupby error
                data_type = self.data_frame[
                    col].cat.categories.dtype if self.data_frame[col].cat.categories.inferred_type \
                        != 'string' else np.str
                self.data_frame[col] = self.data_frame[col].astype(data_type)
        return self.data_frame

    def __exit__(self, vtype, value, traceback):
        if self.as_code:
            for col in self.categorical_cols:
                self.data_frame[col] = pd.Categorical.from_codes(
                    self.data_frame[col], self.categories_dict[col])
        else:
            for col in self.categorical_cols:
                self.data_frame[col] = self.data_frame[col].astype('category')
                self.data_frame[col].cat.set_categories(
                    self.categories_dict[col], inplace=True)


def replace_values_in_cat_col(df: pd.DataFrame, col, subs, new_categories):
    """
    This function replaces values in a categorical column, adding new categories as needed.
    """
    df[col].cat.add_categories(new_categories, inplace=True)
    with Decategorise(df) as decategorised_df:
        decategorised_df[col].replace(subs, inplace=True)


def replace_by_count_cutoff(df: pd.DataFrame,
                            cols,
                            cnt_cutoff,
                            pos_cutoff,
                            invalid_dict=None):
    """
    This function replaces categorical values with too few occurrences with '(other)' and
    returns a dict with the format {"column_name": ["replaced value in column", ...], ...}
    This dict can also be supplied to the function to allow replication of the same replacement, e.g.
    replace the same values in test set as in the training set.
    """
    reset_dict = invalid_dict is None
    if reset_dict:
        invalid_dict = {}
    for col in cols:
        if reset_dict:
            unique_counts = df[col].value_counts()
            pos_counts = df.groupby(col)['totals.transactionRevenue'].apply(
                lambda x: np.sum(x > 0))
            invalid_vals_cnt = set(
                unique_counts[unique_counts <= cnt_cutoff].index)
            invalid_vals_pos = set(pos_counts[pos_counts <= pos_cutoff].index)
            invalids = invalid_vals_cnt.union(invalid_vals_pos)
            invalid_dict[col] = invalids
        else:
            invalids = invalid_dict[col]
        replace_values_in_cat_col(df, col, {c: '(other)'
                                            for c in invalids}, '(other)')
    return invalid_dict


def add_missingness_cols(df: pd.DataFrame, source_data: pd.DataFrame,
                         cols_to_add):
    """
    This function adds missingness columns to the dataframe.
    """
    for col_name in cols_to_add:
        df['miss_' + col_name] = pd.isna(source_data[col_name])


def add_time_group_cols(df: pd.DataFrame, source_data: pd.DataFrame,
                        time_attrs):
    """
    This function extracts time elements from the visitStartTime field and adds them to the dataframe.
    """
    for time_attr in time_attrs:
        df[time_attr] = getattr(source_data['visitStartTime'].dt,
                                time_attr).astype('category')


def add_aggregated_cols(df: pd.DataFrame,
                        fixed_groups,
                        flexible_groups,
                        attr_func_mapping,
                        agg_funcs,
                        prefix=''):
    """
    This function allows data to be aggregated according to a custom rule defined by the user.
    Data will be aggregated by all columns in 'fixed_groups' as well as one column from 'flexible_groups',
    and for each value column in 'attr_func_mapping', the corresponding aggregation functions will be applied.
    For each group-value-function combination of the above, a new column will be added using the transform
    operation.
    """
    with Decategorise(df) as decategorised_df:
        if flexible_groups is None:
            for val_attr, attr_agg_funcs in attr_func_mapping.items():
                for func_name in attr_agg_funcs:
                    decategorised_df['{0}_{1}_{2}'.format(
                        prefix, val_attr,
                        func_name)] = decategorised_df.groupby(fixed_groups)[
                            val_attr].transform(agg_funcs[func_name]).fillna(0)
        else:
            for flexible_group in flexible_groups:
                for val_attr, attr_agg_funcs in attr_func_mapping.items():
                    for func_name in attr_agg_funcs:
                        decategorised_df['{0}_{1}_{2}_{3}'.format(
                            prefix, flexible_group, val_attr,
                            func_name)] = decategorised_df.groupby(
                                fixed_groups +
                                [flexible_group])[val_attr].transform(
                                    agg_funcs[func_name]).fillna(0)


def input_target_extract(df: pd.DataFrame):
    X = df[[
        c for c in df.columns if c not in [
            'fullVisitorId', 'sessionId', 'visitId',
            'totals.transactionRevenue'
        ]
    ]]
    ids = df['fullVisitorId']
    if 'totals.transactionRevenue' in df.columns:
        y = df['totals.transactionRevenue']
        return ids, X, y
    else:
        return ids, X


############################################################################
# read datasets
print('reading datasets...')
train_source_data = pd.read_pickle(path.join(ROOT, TRAIN_FNAME))
train = train_source_data.copy()
test_source_data = pd.read_pickle(path.join(ROOT, TEST_FNANE))
test = test_source_data.copy()

# remove categories with too few values (on training set first and record the changes)
print('pruning infrequent categories...')
invalids_dict = replace_by_count_cutoff(
    train,
    ['device.browser', 'device.operatingSystem', 'trafficSource.source'
     ] + GEO_HIERARCHY[:-1], COUNT_CUTOFF, POS_CLASS_CUTOFF)
# now replicate the process on test set
replace_by_count_cutoff(
    test, ['device.browser', 'device.operatingSystem', 'trafficSource.source']
    + GEO_HIERARCHY[:-1],
    COUNT_CUTOFF,
    POS_CLASS_CUTOFF,
    invalid_dict=invalids_dict)
# replace geo columns' NA with (other) so label-less subgroups can still share one parent group
for col in GEO_HIERARCHY[:-1]:
    train[col].fillna('(other)', inplace=True)
    test[col].fillna('(other)', inplace=True)

# drop columns
print('dropping columns...')
train.drop(COLS_TO_DROP, axis=1, inplace=True)
test.drop(COLS_TO_DROP, axis=1, inplace=True)

# add interaction features
print('adding composite features...')
train['hit_rate'] = train['totals.hits'] / train['totals.pageviews']
test['hit_rate'] = test['totals.hits'] / test['totals.pageviews']

# add missingness columns
print('adding missingness columns...')
add_missingness_cols(train, train_source_data, MISSINGNESS_COLUMNS_TO_ADD)
add_missingness_cols(test, test_source_data, MISSINGNESS_COLUMNS_TO_ADD)

# break down time
print('adding broken-down time columns...')
add_time_group_cols(train, train_source_data, TIMEGROUP_EXTRACTION_LIST)
add_time_group_cols(test, test_source_data, TIMEGROUP_EXTRACTION_LIST)

# add geo-temperal interactions
print('adding geo-temperal interactions...')
add_aggregated_cols(
    train,
    GEO_LEVEL_AGG_LIST,
    GEO_TEMPERAL_TIME_GROUPS,
    GEO_TEMPERAL_ATTR_DICT,
    AGG_FUNC_DICT,
    prefix="geo")
add_aggregated_cols(
    test,
    GEO_LEVEL_AGG_LIST,
    GEO_TEMPERAL_TIME_GROUPS,
    GEO_TEMPERAL_ATTR_DICT,
    AGG_FUNC_DICT,
    prefix="geo")

# add user-level aggregations
print('adding user-level interactions...')
add_aggregated_cols(
    train, ['fullVisitorId'],
    None,
    USER_AGG_ATTR_DICT,
    AGG_FUNC_DICT,
    prefix='user')
add_aggregated_cols(
    test, ['fullVisitorId'],
    None,
    USER_AGG_ATTR_DICT,
    AGG_FUNC_DICT,
    prefix='user')

# add domain-level aggregations
print('adding domain-level interactions...')
add_aggregated_cols(
    train, ['geoNetwork.networkDomain'],
    None,
    DOMAIN_AGG_ATTR_DICT,
    AGG_FUNC_DICT,
    prefix='domain')
add_aggregated_cols(
    test, ['geoNetwork.networkDomain'],
    None,
    DOMAIN_AGG_ATTR_DICT,
    AGG_FUNC_DICT,
    prefix='domain')

###############################################################################
###############################################################################
# split and output sets
# split train and dev by id, ensuring that the same user ends up in the same set
split_ratio = 0.8

unique_ids = train['fullVisitorId'].unique()
np.random.seed(7777)
np.random.shuffle(unique_ids)
N_train = int(len(unique_ids) * split_ratio)
train_ids, dev_ids = unique_ids[:N_train], unique_ids[N_train:]
true_train = train.loc[train['fullVisitorId'].isin(train_ids), :]
dev = train.loc[train['fullVisitorId'].isin(dev_ids), :]

# split set
id_full, X_full, y_full = input_target_extract(train)
id_train, X_train, y_train = input_target_extract(true_train)
id_dev, X_dev, y_dev = input_target_extract(dev)
id_test, X_test = input_target_extract(test)

# sample output
print('-' * 40)
print('X_full dimensions: {0}'.format(X_full.shape))
print('X_train dimensions: {0}'.format(X_train.shape))
print('X_dev dimensions: {0}'.format(X_dev.shape))
print('X_test dimensions: {0}'.format(X_test.shape))
print('-' * 40)
print('Sample of input data:')
print(X_train.head(3).T)
print('-' * 40)

# output sets
X_full.to_pickle(path.join(OUTPUT, 'session_X_full.pkl'))
y_full.to_pickle(path.join(OUTPUT, 'session_y_full.pkl'))
id_full.to_pickle(path.join(OUTPUT, 'session_id_full.pkl'))

X_train.to_pickle(path.join(OUTPUT, 'session_X_train.pkl'))
y_train.to_pickle(path.join(OUTPUT, 'session_y_train.pkl'))
id_train.to_pickle(path.join(OUTPUT, 'session_id_train.pkl'))

X_dev.to_pickle(path.join(OUTPUT, 'session_X_dev.pkl'))
y_dev.to_pickle(path.join(OUTPUT, 'session_y_dev.pkl'))
id_dev.to_pickle(path.join(OUTPUT, 'session_id_dev.pkl'))

X_test.to_pickle(path.join(OUTPUT, 'session_X_test.pkl'))
id_test.to_pickle(path.join(OUTPUT, 'session_id_test.pkl'))

# also output csv as reference 
# (you do lose the category encodings this way)
X_full.to_csv(path.join(OUTPUT, 'session_X_full.csv'), index=None)
y_full.to_csv(path.join(OUTPUT, 'session_y_full.csv'), index=None)
id_full.to_csv(path.join(OUTPUT, 'session_id_full.csv'), index=None)

X_test.to_csv(path.join(OUTPUT, 'session_X_test.csv'), index=None)
id_test.to_csv(path.join(OUTPUT, 'session_id_test.csv'), index=None)

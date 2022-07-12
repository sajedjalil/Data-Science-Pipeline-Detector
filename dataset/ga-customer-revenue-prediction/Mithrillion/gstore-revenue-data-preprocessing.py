import numpy as np
import pandas as pd
import json
from os import path

ROOT = '../input/'
OUTPUT = '.'

processed = {}


# this custom flatten function seems to run faster than json_normalize from Pandas
def json_flatten(obj, parent=""):
    out = {}
    for k, v in obj.items():
        full_key = "{0}.{1}".format(parent, k) if parent != "" else k
        if type(v) != dict:
            out[full_key] = v
        else:
            out = {**out, **json_flatten(v, parent=full_key)}
    return out


def flatten_col(col):
    return pd.DataFrame(
        [json_flatten(json.loads(r), parent=col.name) for r in col])


for set_name in ["train", "test"]:
    # read data file
    dat = pd.read_csv(
        path.join(ROOT, "{0}.csv".format(set_name)),
        dtype={'fullVisitorId': np.str})

    # flatten json columns
    flat = pd.concat(
        list(
            map(flatten_col, [
                dat['device'], dat['geoNetwork'], dat['totals'],
                dat['trafficSource']
            ])),
        axis=1)

    # merge json and non-json columns
    full = pd.concat(
        [
            dat.loc[:, ~dat.columns.isin(
                ['device', 'geoNetwork', 'totals', 'trafficSource'])], flat
        ],
        axis=1)

    # replace various missing values with NaN
    full.replace(
        {
            '(not set)': np.nan,
            'not available in demo dataset': np.nan,
            '(not provided)': np.nan,
            'unknown.unknown': np.nan
        },
        inplace=True)

    # exclude columns with only one value (incl. N/A)
    desc = full.describe(include='all').T
    single_val_cols = desc.index[(desc['count'] <= 1) | (
        (desc['unique'] == 1) & (desc['count'] == len(dat)))]
    valid = full.loc[:, ~full.columns.isin(single_val_cols)].copy()
    del full, dat, flat

    # convert date/time columns to datetime type
    date_col = pd.to_datetime(valid.date, format='%Y%m%d')
    start_col = pd.to_datetime(valid.visitStartTime, unit='s')
    valid.loc[:, 'date'] = date_col
    valid.loc[:, 'visitStartTime'] = start_col

    # convert numerical columns to float (as Python int does not accept NA) and binary to boolean
    if set_name == "train":
        valid['totals.transactionRevenue'].replace(np.nan, 0, inplace=True)
        valid['totals.transactionRevenue'] = valid[
            'totals.transactionRevenue'].astype(np.float)
    valid['totals.hits'] = valid['totals.hits'].astype(np.float)
    valid['totals.pageviews'] = valid['totals.pageviews'].astype(np.float)
    valid['totals.bounces'] = valid['totals.bounces'].astype(np.float)
    valid['totals.newVisits'] = valid['totals.newVisits'].astype(np.float)
    valid['trafficSource.adwordsClickInfo.isVideoAd'] = valid[
        'trafficSource.adwordsClickInfo.isVideoAd'].astype(np.bool)
    valid['trafficSource.isTrueDirect'] = valid[
        'trafficSource.isTrueDirect'].astype(np.bool)
    valid['trafficSource.adwordsClickInfo.page'] = valid[
        'trafficSource.adwordsClickInfo.page'].astype(np.float)

    processed[set_name] = valid

# convert categorical columns to category type (for compression and ease of encoding with integers)
category_col_list = [
    'channelGrouping', 'device.browser', 'device.deviceCategory',
    'device.operatingSystem', 'geoNetwork.city', 'geoNetwork.continent',
    'geoNetwork.country', 'geoNetwork.metro', 'geoNetwork.region',
    'geoNetwork.networkDomain', 'geoNetwork.subContinent',
    'trafficSource.adContent', 'trafficSource.adwordsClickInfo.adNetworkType',
    'trafficSource.adwordsClickInfo.slot', 'trafficSource.campaign',
    'trafficSource.medium', 'trafficSource.source'
]

# ensure the training and testing set have the same categorical encodings
for col_name in category_col_list:
    for set_name in ["train", "test"]:
        processed[set_name][col_name] = processed[set_name][col_name].astype(
            'category')
    mapping_set = set(processed['train'][col_name].cat.categories).union(
        set(processed['test'][col_name].cat.categories))
    for set_name in ["train", "test"]:
        processed[set_name][col_name].cat.set_categories(
            list(mapping_set), inplace=True)

for set_name in ["train", "test"]:
    processed[set_name].to_pickle(
        path.join(OUTPUT, "{0}.pkl".format(set_name)))

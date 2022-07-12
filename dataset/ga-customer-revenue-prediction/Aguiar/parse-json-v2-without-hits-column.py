# GA CUSTOMER REVENUE COMPETITION
# Updated kernel (11/11) with v2 files
# Read and preprocess all columns, except hits.

import gc
import os
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import json
import time
from ast import literal_eval


def load_df(file_name = 'train_v2.csv', nrows = None):
    """Read csv and convert json columns."""
    
    USE_COLUMNS = [
        'channelGrouping', 'date', 'device', 'fullVisitorId', 'geoNetwork',
        'socialEngagementType', 'totals', 'trafficSource', 'visitId',
        'visitNumber', 'visitStartTime', 'customDimensions',
        #'hits'
    ]

    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    df = pd.read_csv('../input/{}'.format(file_name),
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, nrows=nrows, usecols=USE_COLUMNS)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        
    # Normalize customDimensions
    df['customDimensions']=df['customDimensions'].apply(literal_eval)
    df['customDimensions']=df['customDimensions'].str[0]
    df['customDimensions']=df['customDimensions'].apply(lambda x: {'index':np.NaN,'value':np.NaN} if pd.isnull(x) else x)

    column_as_df = json_normalize(df['customDimensions'])
    column_as_df.columns = [f"customDimensions_{subcolumn}" for subcolumn in column_as_df.columns]
    df = df.drop('customDimensions', axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df

    
def pipeline():
    timer = time.time()
    train = load_df('train_v2.csv')
    # Drop constant columns in train and test
    const_cols = [c for c in train.columns if train[c].nunique(dropna=False) < 2]
    const_cols.append('customDimensions_index')  # Also not usefull
    train.drop(const_cols, axis=1, inplace=True)
    # Drop campaignCode (has only 1 example that is not NaN) - only on train set
    train.drop('trafficSource_campaignCode', axis=1, inplace=True)
    # Save as pickle file (could be hdf5 or feather too)
    train.to_pickle('train_v2_clean.pkl')
    print("Train shape", train.shape)
    del train; gc.collect()
    
    test = load_df('test_v2.csv')
    # Drop constant columns in train
    test.drop(const_cols, axis=1, inplace=True)
    # Save as pickle file (could be hdf5 or feather too)
    test.to_pickle('test_v2_clean.pkl')
    print("Test shape", test.shape)
    print("Pipeline completed in {}s".format(time.time() - timer))
    
pipeline()
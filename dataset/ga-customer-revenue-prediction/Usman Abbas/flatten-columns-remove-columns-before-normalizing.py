#use pd.read_hdf() to read hdf files

import numpy as np 
import pandas as pd 
import json
import bq_helper
from pandas.io.json import json_normalize
import seaborn as sns 
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error
import functools
from multiprocessing import Pool
import logging
import gc
import logging
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error
from scipy.stats import stats
from ast import literal_eval
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

def parse(csv_path='../input/train_v2.csv', nrows=None):

    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)

    device_list=df['device'].tolist()
    
    #deleting unwanted columns before normalizing
    for device in device_list:
        del device['browserVersion'],device['browserSize'],device['flashVersion'],device['mobileInputSelector'],device['operatingSystemVersion'],device['screenResolution'],device['screenColors']
    df['device']=pd.Series(device_list)
    
    geoNetwork_list=df['geoNetwork'].tolist()
    for network in geoNetwork_list:
        del network['latitude'],network['longitude'],network['networkLocation'],network['cityId']
    df['geoNetwork']=pd.Series(geoNetwork_list)
    
    df['hits']=df['hits'].apply(literal_eval)
    df['hits']=df['hits'].str[0]
    to_replace_keys=['time',
     'hour',
     'minute',
     'isInteraction',
     'isEntrance',
     'isExit',
     'referer',
     'page',
     'transaction',
     'item',
     'appInfo',
     'exceptionInfo',
     'product',
     'promotion',
     'eCommerceAction',
     'experiment',
     'customVariables',
     'customDimensions',
     'customMetrics',
     'type',
     'social',
     'contentGroup',
     'dataSource',
     'publisher_infos']
    
    to_replace={key:np.NaN for key in to_replace_keys}
    df['hits']=df['hits'].apply(lambda x: to_replace if pd.isnull(x) else x)
    
    df['customDimensions']=df['customDimensions'].apply(literal_eval)
    df['customDimensions']=df['customDimensions'].str[0]
    df['customDimensions']=df['customDimensions'].apply(lambda x: {'index':np.NaN,'value':np.NaN} if pd.isnull(x) else x)
    
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource','hits','customDimensions']

    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    
    return df


def load_data(nrows=None):
    train_df = parse(nrows=nrows)
    test_df = parse("../input/test_v2.csv",nrows)
    return train_df,test_df


train_df,test_df=load_data(50000)
print(train_df.shape,test_df.shape)

train_df.to_hdf('train_df.h5','data')
test_df.to_hdf('test_df.h5','data')
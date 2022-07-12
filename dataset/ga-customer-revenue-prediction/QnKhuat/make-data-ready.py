import gc
import os
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import json
import time
from tqdm import tqdm
from ast import literal_eval

def to_na(df):
    # Each type of columns that need to replace with the right na values
    to_NA_cols = ['trafficSource_adContent','trafficSource_adwordsClickInfo.adNetworkType',
                'trafficSource_adwordsClickInfo.slot','trafficSource_adwordsClickInfo.gclId',
                'trafficSource_keyword','trafficSource_referralPath','customDimensions_value']

    to_0_cols = ['totals_transactionRevenue','trafficSource_adwordsClickInfo.page','totals_sessionQualityDim','totals_bounces',
                 'totals_timeOnSite','totals_newVisits','totals_pageviews','customDimensions_index','totals_transactions','totals_totalTransactionRevenue']

    to_true_cols = ['trafficSource_adwordsClickInfo.isVideoAd']
    to_false_cols = ['trafficSource_isTrueDirect']
    
    
    df[to_NA_cols] = df[to_NA_cols].fillna('NA')
    df[to_0_cols] = df[to_0_cols].fillna(0)
    df[to_true_cols] = df[to_true_cols].fillna(True)
    df[to_false_cols] = df[to_false_cols].fillna(False)
    
    return df
    
def encode_date(df):
    fld = pd.to_datetime(df['date'], infer_datetime_format=True)
    
    attrs = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
        'Is_month_end', 'Is_month_start', 'Is_quarter_end', 
        'Is_quarter_start', 'Is_year_end', 'Is_year_start','Hour']
        
    for attr in attrs:
        df['Date_'+attr] = getattr(fld.dt,attr.lower())
        
    return df

def weird_na(df):
    cols_to_replace = {
        'socialEngagementType' : 'Not Socially Engaged',
        'device_browserSize' : 'not available in demo dataset', 
        'device_flashVersion' : 'not available in demo dataset', 
        'device_browserVersion' : 'not available in demo dataset', 
        'device_language' : 'not available in demo dataset',
        'device_mobileDeviceBranding' : 'not available in demo dataset',
        'device_mobileDeviceInfo' : 'not available in demo dataset',
        'device_mobileDeviceMarketingName' : 'not available in demo dataset',
        'device_mobileDeviceModel' : 'not available in demo dataset',
        'device_mobileInputSelector' : 'not available in demo dataset',
        'device_operatingSystemVersion' : 'not available in demo dataset',
        'device_screenColors' : 'not available in demo dataset',
        'device_screenResolution' : 'not available in demo dataset',
        'geoNetwork_city' : 'not available in demo dataset',
        'geoNetwork_cityId' : 'not available in demo dataset',
        'geoNetwork_latitude' : 'not available in demo dataset',
        'geoNetwork_longitude' : 'not available in demo dataset',
        'geoNetwork_metro' : ['not available in demo dataset', '(not set)'], 
        'geoNetwork_networkDomain' : 'unknown.unknown', 
        'geoNetwork_networkLocation' : 'not available in demo dataset',
        'geoNetwork_region' : 'not available in demo dataset',
        'trafficSource_adwordsClickInfo.criteriaParameters' : 'not available in demo dataset',
        'trafficSource_campaign' : '(not set)', 
        'trafficSource_keyword' : '(not provided)',
        'networkDomain': '(not set)', 
        'city': '(not set)', 
    }
    df = df.replace(cols_to_replace,'NA')
    return df

def del_const(df):
    const_col = []
    for col in df.columns:
        if df[col].nunique() == 1 and df[col].isnull().sum()==0 :
            const_col.append(col)
            
    df.drop(const_col,axis=1,inplace=True)
    return df, const_col
    
def json_it(df):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
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

def convert_it(df):
    # convert weird string to na
    df = weird_na(df)
    
    # Convert columns to Na on it own type
    df = to_na(df)
    
    # create new columsn with data
    df_train = encode_date(df)
    
    return df
    
def fix_type(df):
    try:
        df.drop('trafficSource_campaignCode',axis=1,inplace=True)
    except:
        pass
    # Fill na and rename the Revenue column
    df['totals_transactionRevenue'] = df['totals_transactionRevenue'].fillna(0).astype(float)

    to_int = ['totals_bounces','totals_newVisits','totals_pageviews',
            'customDimensions_index','totals_hits','totals_sessionQualityDim',
            'totals_visits','totals_timeOnSite','trafficSource_adwordsClickInfo.page',
            'totals_transactions','totals_totalTransactionRevenue']
    for col in to_int :
        df[col] = df[col].astype(int)

    return df
    
def load_it(csv_path,name):
    CONST_COLLUMNS = ['socialEngagementType','device_browserSize',
         'device_browserVersion','device_flashVersion',
         'device_language','device_mobileDeviceBranding',
         'device_mobileDeviceInfo','device_mobileDeviceMarketingName',
         'device_mobileDeviceModel','device_mobileInputSelector',
         'device_operatingSystemVersion','device_screenColors',
         'device_screenResolution','geoNetwork_cityId',
         'geoNetwork_latitude','geoNetwork_longitude',
         'geoNetwork_networkLocation',
         'trafficSource_adwordsClickInfo.criteriaParameters',]
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    dfs = pd.read_csv(csv_path, sep=',',
                    parse_dates=['date'],
                    converters={column: json.loads for column in JSON_COLUMNS}, 
                    dtype={'fullVisitorId': 'str'}, # Important!!
                    chunksize = 200000)
    
    for idx,df in enumerate(dfs):
        print(idx)
        df.reset_index(drop = True,inplace = True)
        df = json_it(df)
        df = convert_it(df)
        df.drop(CONST_COLLUMNS,axis=1,inplace=True)
        # Heavy as hell this column
        df.drop('hits',axis=1,inplace=True)
        df = fix_type(df)
        df.to_pickle(f'{name}_{idx}.pkl')
        
        del df
        gc.collect()
    print('Done')
  
  
load_it('../input/train_v2.csv','train')

load_it('../input/test_v2.csv','test')
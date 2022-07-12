import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import time
from datetime import datetime
import gc
import psutil
from sklearn.preprocessing import LabelEncoder
import pickle

fe = True

if fe:
    PATH="../input/"
    NUM_ROUNDS = 20000
    VERBOSE_EVAL = 500
    STOP_ROUNDS = 100
    N_SPLITS = 10
    
     #the columns that will be parsed to extract the fields from the jsons
    cols_to_parse = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    def read_parse_dataframe(file_name):
        #full path for the data file
        path = PATH + file_name
        #read the data file, convert the columns in the list of columns to parse using json loader,
        #convert the `fullVisitorId` field as a string
        data_df = pd.read_csv(path, 
            converters={column: json.loads for column in cols_to_parse}, 
            dtype={'fullVisitorId': 'str'})
        #parse the json-type columns
        for col in cols_to_parse:
            #each column became a dataset, with the columns the fields of the Json type object
            json_col_df = json_normalize(data_df[col])
            json_col_df.columns = [f"{col}_{sub_col}" for sub_col in json_col_df.columns]
            #we drop the object column processed and we add the columns created from the json fields
            data_df = data_df.drop(col, axis=1).merge(json_col_df, right_index=True, left_index=True)
        return data_df
        
    def process_date_time(data_df):
        print("process date time ...")
        data_df['date'] = data_df['date'].astype(str)
        data_df["date"] = data_df["date"].apply(lambda x : x[:4] + "-" + x[4:6] + "-" + x[6:])
        data_df["date"] = pd.to_datetime(data_df["date"])   
        data_df["year"] = data_df['date'].dt.year
        data_df["month"] = data_df['date'].dt.month
        data_df["day"] = data_df['date'].dt.day
        data_df["weekday"] = data_df['date'].dt.weekday
        data_df['weekofyear'] = data_df['date'].dt.weekofyear
        data_df['month_unique_user_count'] = data_df.groupby('month')['fullVisitorId'].transform('nunique')
        data_df['day_unique_user_count'] = data_df.groupby('day')['fullVisitorId'].transform('nunique')
        data_df['weekday_unique_user_count'] = data_df.groupby('weekday')['fullVisitorId'].transform('nunique')
    
        return data_df
    
    def process_format(data_df):
        print("process format ...")
        for col in ['visitNumber', 'totals_hits', 'totals_pageviews']:
            data_df[col] = data_df[col].astype(float)
        data_df['trafficSource_adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
        data_df['trafficSource_isTrueDirect'].fillna(False, inplace=True)
        return data_df
        
    def process_device(data_df):
        print("process device ...")
        data_df['browser_category'] = data_df['device_browser'] + '_' + data_df['device_deviceCategory']
        data_df['browser_os'] = data_df['device_browser'] + '_' + data_df['device_operatingSystem']
        return data_df
    
    def process_totals(data_df):
        print("process totals ...")
        data_df['visitNumber'] = np.log1p(data_df['visitNumber'])
        data_df['totals_hits'] = np.log1p(data_df['totals_hits'])
        data_df['totals_pageviews'] = np.log1p(data_df['totals_pageviews'].fillna(0))
        data_df['mean_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('mean')
        data_df['sum_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('sum')
        data_df['max_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('max')
        data_df['min_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('min')
        data_df['var_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('var')
        data_df['mean_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('mean')
        data_df['sum_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('sum')
        data_df['max_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('max')
        data_df['min_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('min')    
        return data_df
    
    def process_geo_network(data_df):
        print("process geo network ...")
        data_df['sum_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('sum')
        data_df['count_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('count')
        data_df['mean_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('mean')
        data_df['sum_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('sum')
        data_df['count_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('count')
        data_df['mean_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('mean')
        return data_df
    
    def process_traffic_source(data_df):
        print("process traffic source ...")
        data_df['source_country'] = data_df['trafficSource_source'] + '_' + data_df['geoNetwork_country']
        data_df['campaign_medium'] = data_df['trafficSource_campaign'] + '_' + data_df['trafficSource_medium']
        data_df['medium_hits_mean'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('mean')
        data_df['medium_hits_max'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('max')
        data_df['medium_hits_min'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('min')
        data_df['medium_hits_sum'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('sum')
        return data_df
        
    def custom(data):
        print('custom..')
        data['device_deviceCategory_channelGrouping'] = data['device_deviceCategory'] + "_" + data['channelGrouping']
        data['channelGrouping_browser'] = data['device_browser'] + "_" + data['channelGrouping']
        #data['channelGrouping_mobile'] = str(data['device_isMobile']) + "_" + data['channelGrouping']
        #data['browser_mobile'] = str(data['device_isMobile']) + "_" + data['device_browser']
        data['channelGrouping_OS'] = data['device_operatingSystem'] + "_" + data['channelGrouping']
        
        for i in ['geoNetwork_city', 'geoNetwork_continent', 'geoNetwork_country','geoNetwork_metro', 'geoNetwork_networkDomain', 'geoNetwork_region','geoNetwork_subContinent']:
            for j in ['device_browser','device_deviceCategory', 'device_operatingSystem', 'trafficSource_source']:
                data[i + "_" + j] = data[i] + "_" + data[j]
        
        data['content_source'] = data['trafficSource_adContent'] + "_" + data['source_country']
        data['medium_source'] = data['trafficSource_medium'] + "_" + data['source_country']
        data['max_pageviews_WD'] = data.groupby(['weekday'])['totals_pageviews'].transform('max')
        data['min_pageviews_WD'] = data.groupby(['weekday'])['totals_pageviews'].transform('min')
        return data

    
    #Feature processing
    ## Load data
    print('reading train')
    train_df = read_parse_dataframe('train.csv')
    trn_len = train_df.shape[0]
    train_df = process_date_time(train_df)
    print('reading test')
    test_df = read_parse_dataframe('test.csv')
    test_df = process_date_time(test_df)
    #test_is_bounce_index = test_df[test_df['totals_bounces'].notnull()].index.copy()
    ## Drop columns
    cols_to_drop = [col for col in train_df.columns if train_df[col].nunique(dropna=False) == 1]
    train_df.drop(cols_to_drop, axis=1, inplace=True)
    test_df.drop([col for col in cols_to_drop if col in test_df.columns], axis=1, inplace=True)
    
    ###only one not null value
    train_df.drop(['trafficSource_campaignCode'], axis=1, inplace=True)
    
    ###converting columns format
    train_df['totals_transactionRevenue'] = train_df['totals_transactionRevenue'].astype(float)
    train_df['totals_transactionRevenue'] = train_df['totals_transactionRevenue'].fillna(0)
    train_df['totals_transactionRevenue'] = np.log1p(train_df['totals_transactionRevenue'])
    
    
    ## Features engineering
    train_df = process_format(train_df)
    train_df = process_device(train_df)
    train_df = process_totals(train_df)
    train_df = process_geo_network(train_df)
    train_df = process_traffic_source(train_df)
    train_df = custom(train_df)
    
    test_df = process_format(test_df)
    test_df = process_device(test_df)
    test_df = process_totals(test_df)
    test_df = process_geo_network(test_df)
    test_df = process_traffic_source(test_df)
    test_df = custom(test_df)
    
    ## Categorical columns
    print("process categorical columns ...")
    num_cols = ['month_unique_user_count', 'day_unique_user_count', 'weekday_unique_user_count',
                'visitNumber', 'totals_hits', 'totals_pageviews', 
                'mean_hits_per_day', 'sum_hits_per_day', 'min_hits_per_day', 'max_hits_per_day', 'var_hits_per_day',
                'mean_pageviews_per_day', 'sum_pageviews_per_day', 'min_pageviews_per_day', 'max_pageviews_per_day',
                'sum_pageviews_per_network_domain', 'count_pageviews_per_network_domain', 'mean_pageviews_per_network_domain',
                'sum_hits_per_network_domain', 'count_hits_per_network_domain', 'mean_hits_per_network_domain',
                'medium_hits_mean','medium_hits_min','medium_hits_max','medium_hits_sum']
                
    not_used_cols = ["visitNumber", "date", "fullVisitorId", "sessionId", 
            "visitId", "visitStartTime", 'totals_transactionRevenue', 'trafficSource_referralPath']
    cat_cols = [col for col in train_df.columns if col not in num_cols and col not in not_used_cols]
    
    for col in cat_cols:
        #print(col)
        lbl = LabelEncoder()
        lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
        train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
        test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))
    
    print('FINAL train shape : ', train_df.shape, ' test shape : ', test_df.shape)
    #print(train_df.columns)
    train_df = train_df.sort_values('date')
    X = train_df.drop(not_used_cols, axis=1)
    y = train_df['totals_transactionRevenue']
    #test_is_bounce_index = test_df[test_df['totals_bounces'].notnull()].index.copy()
    X_test = test_df.drop([col for col in not_used_cols if col in test_df.columns], axis=1)

import keras 
from keras.utils import Sequence
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Concatenate, Dropout
from sklearn.model_selection import train_test_split

#X_test['totals_transactionRevenue'] = np.nan
X = X.append(X_test).reset_index()
X = X.fillna(0)
X[np.isnan(X)] = 0
for i in X.columns:
    X[i] = X[i].fillna(0)
#print(X.isnull().any())
del X_test
gc.collect()
print("NaN : ", X.isnull().values.any())

cols = X.shape[1]
xt, xval = train_test_split(X, test_size = 0.1, random_state = 7)

inp = Input(shape=(cols,))
encoded = Dense(cols, activation='relu')(inp)
encoded = Dropout(0.5)(encoded)
encoded = Dense(cols // 2, activation='relu')(encoded)
encoded = Dropout(0.3)(encoded)
encoded = Dense(cols // 3, activation='tanh')(encoded)

decoded = Dense(cols // 3, activation='tanh')(encoded)
decoded = Dropout(0.3)(decoded)
decoded = Dense(cols // 2, activation='relu')(decoded)
decoded = Dropout(0.5)(decoded)
decoded = Dense(cols, activation='relu')(decoded)

autoencoder = Model(inp, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

autoencoder.fit(xt, xt,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(xval, xval),
                verbose = 2)
ans = autoencoder.predict(X)
try:
    ans.to_csv('denoiced.csv', index = False)
except:
    np.save('ans.npy', np.array(ans))

'''
np.save('B', np.array(1))
del dae_denoised_data
gc.collect()
#after you have DAE fitted...
your_new_df=dae_data.copy()

for i in ['1','2','3']:
    print('Hidden layer',i)
    columns_names = ['Hidden_'+str(i)+'_'+str(l) for l in range(0, len_input_columns*2)]
    for l in columns_names:
        your_new_df[l] = 0 
    intermediate_layer_model = Model(inputs=model_dae.input, outputs=model_dae.get_layer('Hidden' + i).output)
    your_new_df[columns_names] = intermediate_layer_model.predict(dae_data)
np.save('DONE', np.array(1)) 
print('DONE!')
try:
    your_new_df.to_csv('your_new_df.csv', index = False)
except:
    #open('your_new_df.pkl', 'w').write(pickle.dumps(your_new_df))
    np.save('your_new_df.npy', np.array(your_new_df))
'''
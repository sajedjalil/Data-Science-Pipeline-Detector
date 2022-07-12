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

#Feature processing
## Load data
print('reading train')
train_df = read_parse_dataframe('train.csv')
trn_len = train_df.shape[0]
train_df = process_date_time(train_df)
print('reading test')
test_df = read_parse_dataframe('test.csv')
test_df = process_date_time(test_df)

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

test_df = process_format(test_df)
test_df = process_device(test_df)
test_df = process_totals(test_df)
test_df = process_geo_network(test_df)
test_df = process_traffic_source(test_df)

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

merged_df = pd.concat([train_df, test_df])
print('Cat columns : ', len(cat_cols))
ohe_cols = []
for i in cat_cols:
    if len(set(merged_df[i].values)) < 100:
        ohe_cols.append(i)

print('ohe_cols : ', ohe_cols)
print(len(ohe_cols))
merged_df = pd.get_dummies(merged_df, columns = ohe_cols)
train_df = merged_df[:trn_len]
test_df = merged_df[trn_len:]
del merged_df
gc.collect()

for col in cat_cols:
    if col in ohe_cols:
        continue
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
X_test = test_df.drop([col for col in not_used_cols if col in test_df.columns], axis=1)
del train_df
gc.collect()
testid = test_df[['fullVisitorId']].copy()
del test_df
gc.collect()
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor

class SklearnWrapper(object):
    def __init__(self, clf, seed=7, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        print("Training..")
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        print("Predicting..")
        return self.clf.predict(x)

from sklearn.cross_validation import KFold
kf = KFold(X.shape[0], n_folds=5, shuffle=True, random_state=7)
NFOLDS = 5
def get_oof(clf):
    oof_train = np.zeros(len(X))
    oof_test = np.zeros(len(X_test))
    # oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = X.iloc[train_index].values
        y_tr = y.iloc[train_index].values
        x_te = X.iloc[test_index].values
        y_te = y.iloc[test_index].values
        
        print("train ", i, "-"*50)
        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)  
        cv = mean_squared_error(oof_train[test_index], y_te)**0.5
        cv = str(cv)
        cv = cv[:10]
        print("Fold ", (i + 1), " CV : ", cv)
        oof_test += clf.predict(X_test) / NFOLDS
        np.save(str(i),np.array(i))
        #pd.DataFrame({'oof_preds': oof_train[test_index]}).to_csv('rf_oof_fold' + str(i) + "_" + str(cv) + '.csv', index = False)
        #pd.DataFrame({'ans_preds': oof_test}).to_csv('rf_ans_fold' + str(i) + "_" + str(cv) + '.csv', index = False)
    #pd.DataFrame({'oof_preds': oof_train}).to_csv('rf_oof.csv', index = False)
    return oof_train, oof_test

rf_params = {
    'n_estimators': 100, 
    'n_jobs': -1, 
    'verbose': 1, 
    'max_depth': 10,
    'min_samples_split': 5, 
    'random_state': 7
    }

rf = SklearnWrapper(clf=RandomForestRegressor, seed=7, params=rf_params)
print("rf..")
rfoof, rfans = get_oof(rf)
ans_rf = []
for i in rfans:
    if i < 0:
        ans_rf.append(0)
    else:
        ans_rf.append(i)
o_rf = []
for i in rfoof:
    if i < 0:
        o_rf.append(0)
    else:
        o_rf.append(i)
del rfoof, rfans
gc.collect()

cvrf = mean_squared_error(o_rf, y)**0.5
cvrf = str(cvrf)
cvrf = cvrf[:10]
pd.DataFrame({'preds': o_rf}).to_csv('rf_oof_' + cvrf + '.csv', index = False)
submission = testid
submission.loc[:, 'PredictedLogRevenue'] = ans_rf
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0.0)
grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test.to_csv('rf_' + cvrf + '.csv',index=False)
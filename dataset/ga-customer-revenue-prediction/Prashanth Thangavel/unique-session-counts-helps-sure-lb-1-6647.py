## Hi guys, this is an extension of Gabriel Preda
## Whats new? 
## Added unique session counts in 
## a) Day
## b) Weekday
## c) Month

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

IS_LOCAL=False
if(IS_LOCAL):
    PATH="../input/google-analytics-customer-revenue-prediction/"    
else:
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
    data_df['month_unique_s_count'] = data_df.groupby('month')['sessionId'].transform('nunique')
    data_df['day_unique_s_count'] = data_df.groupby('day')['sessionId'].transform('nunique')
    data_df['weekday_unique_s_count'] = data_df.groupby('weekday')['sessionId'].transform('nunique')
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
    data_df['browser_operatingSystem'] = data_df['device_browser'] + '_' + data_df['device_operatingSystem']
    data_df['source_country'] = data_df['trafficSource_source'] + '_' + data_df['geoNetwork_country']
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


#Feature processing
## Load data
train_df = read_parse_dataframe('train.csv')
train_df = process_date_time(train_df)
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

test_df = process_format(test_df)
test_df = process_device(test_df)
test_df = process_totals(test_df)
test_df = process_geo_network(test_df)

## Categorical columns
print("process categorical columns ...")
num_cols = ['month_unique_user_count', 'day_unique_user_count', 'weekday_unique_user_count',
            'month_unique_s_count', 'day_unique_s_count', 'weekday_unique_s_count',
            'visitNumber', 'totals_hits', 'totals_pageviews', 
            'mean_hits_per_day', 'sum_hits_per_day', 'min_hits_per_day', 'max_hits_per_day', 'var_hits_per_day',
            'sum_pageviews_per_network_domain', 'count_pageviews_per_network_domain', 'mean_pageviews_per_network_domain',
            'sum_hits_per_network_domain', 'count_hits_per_network_domain', 'mean_hits_per_network_domain']
            
not_used_cols = ["visitNumber", "date", "fullVisitorId", "sessionId", 
        "visitId", "visitStartTime", 'totals_transactionRevenue', 'trafficSource_referralPath']
cat_cols = [col for col in train_df.columns if col not in num_cols and col not in not_used_cols]
for col in cat_cols:
    print(col)
    lbl = LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

# Model
print("prepare model ...")
train_df = train_df.sort_values('date')
X = train_df.drop(not_used_cols, axis=1)
y = train_df['totals_transactionRevenue']
X_test = test_df.drop([col for col in not_used_cols if col in test_df.columns], axis=1)

## Model parameters
params = {"objective" : "regression", "metric" : "rmse", "max_depth": 8, "min_child_samples": 20, "reg_alpha": 1, "reg_lambda": 1,
        "num_leaves" : 257, "learning_rate" : 0.01, "subsample" : 0.8, "colsample_bytree" : 0.8, "subsample_freq ": 5}
folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
## Model
print("model ...")
model = lgb.LGBMRegressor(**params, n_estimators = NUM_ROUNDS, nthread = 4, n_jobs = -1)

prediction = np.zeros(test_df.shape[0])

for fold_n, (train_index, test_index) in enumerate(folds.split(X)):
    print('Fold:', fold_n)
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',
            verbose=VERBOSE_EVAL, early_stopping_rounds=STOP_ROUNDS)
    
    y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
    prediction += y_pred
prediction /= N_SPLITS

# Submission
print("prepare submission ...")
submission = test_df[['fullVisitorId']].copy()
submission.loc[:, 'PredictedLogRevenue'] = prediction
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0.0)
grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test.to_csv('submission.csv',index=False)
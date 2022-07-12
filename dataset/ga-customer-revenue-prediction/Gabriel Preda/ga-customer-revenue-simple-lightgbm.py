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


IS_LOCAL=False
if(IS_LOCAL):
    PATH="../input/google-analytics-customer-revenue-prediction/"    
else:
    PATH="../input/"
 
NUM_ROUNDS = 20000
VERBOSE_EVAL = 250
STOP_ROUNDS = 250
N_SPLITS = 9
TIME_SPLIT = True

def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger


logger = get_logger()
 #the columns that will be parsed to extract the fields from the jsons
cols_to_parse = ['device', 'geoNetwork', 'totals', 'trafficSource']


def read_parse_dataframe(file_name=None, nrows=None):
    logger.info('Start read parse')
    #full path for the data file
    path = PATH + file_name
    #read the data file, convert the columns in the list of columns to parse using json loader,
    #convert the `fullVisitorId` field as a string
    data_df = pd.read_csv(path, 
        converters={column: json.loads for column in cols_to_parse}, 
        dtype={'fullVisitorId': 'str'}, 
        nrows=nrows)
    #parse the json-type columns
    for col in cols_to_parse:
        #each column became a dataset, with the columns the fields of the Json type object
        json_col_df = json_normalize(data_df[col])
        json_col_df.columns = [f"{col}_{sub_col}" for sub_col in json_col_df.columns]
        #we drop the object column processed and we add the columns created from the json fields
        data_df = data_df.drop(col, axis=1).merge(json_col_df, right_index=True, left_index=True)
    logger.info('Done with read parse')
    return data_df
    
def process_date_time(data_df):
    logger.info('Start date')
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
    data_df['weekofyear_unique_user_count'] = data_df.groupby('weekofyear')['fullVisitorId'].transform('nunique')
    logger.info('Done with date')
    
    return data_df

def process_format(data_df):
    logger.info('Start format')
    for col in ['visitNumber', 'totals_hits', 'totals_pageviews']:
        data_df[col] = data_df[col].astype(float)
    data_df['trafficSource_adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
    data_df['trafficSource_isTrueDirect'].fillna(False, inplace=True)
    logger.info('Done with format')
    return data_df
    
def process_device(data_df):
    logger.info('Start device')
    data_df['browser_category'] = data_df['device_browser'] + '_' + data_df['device_deviceCategory']
    data_df['browser_os'] = data_df['device_browser'] + '_' + data_df['device_operatingSystem']
    data_df['category_os'] = data_df['device_deviceCategory'] + '_' + data_df['device_operatingSystem']
    data_df['visits_id_browser_mean'] = np.log1p(data_df.groupby(['device_browser'])['visitNumber'].transform('mean'))
    data_df['visits_id_os_mean'] = np.log1p(data_df.groupby(['device_operatingSystem'])['visitNumber'].transform('mean'))
    data_df['visits_id_cat_mean'] = np.log1p(data_df.groupby(['device_deviceCategory'])['visitNumber'].transform('mean'))
    data_df['browser_unique_user_count'] = data_df.groupby('device_browser')['fullVisitorId'].transform('nunique')
    data_df['os_unique_user_count'] = data_df.groupby('device_operatingSystem')['fullVisitorId'].transform('nunique')
    logger.info('Done with device')
    return data_df

def process_totals(data_df):
    logger.info('Start totals')
    #data_df['visitNumber'] = data_df['visitNumber']
    data_df['visits_id_sum'] = data_df.groupby(['fullVisitorId'])['visitNumber'].transform('sum')
    data_df['visits_id_min'] =  data_df.groupby(['fullVisitorId'])['visitNumber'].transform('min')
    data_df['visits_id_max'] = data_df.groupby(['fullVisitorId'])['visitNumber'].transform('max')
    data_df['visits_id_mean'] = data_df.groupby(['fullVisitorId'])['visitNumber'].transform('mean')
    data_df['visits_id_nunique'] = data_df.groupby('visitNumber')['fullVisitorId'].transform('nunique')
    #data_df['totals_hits'] = data_df['totals_hits']
    data_df['hits_id_sum'] = data_df.groupby(['fullVisitorId'])['totals_hits'].transform('sum')
    data_df['hits_id_cnt'] = data_df.groupby(['fullVisitorId'])['totals_hits'].transform('count')
    data_df['totals_pageviews'] = data_df['totals_pageviews'].fillna(0)
    data_df['pageviews_id_sum'] = data_df.groupby(['fullVisitorId'])['totals_pageviews'].transform('sum')
    data_df['pageviews_id_cnt'] = data_df.groupby(['fullVisitorId'])['totals_pageviews'].transform('count')
    data_df['mean_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('mean')
    data_df['sum_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('sum')
    data_df['max_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('max')
    data_df['min_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('min')
    data_df['var_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('var')
    data_df['mean_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('mean')
    data_df['sum_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('sum')
    data_df['max_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('max')
    data_df['min_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('min')   
    
    logger.info('Done with totals')
    return data_df

def process_geo_network(data_df):
    logger.info('Start geoNetworks')
    data_df['mean_visit_per_network_domain']= data_df.groupby('geoNetwork_networkDomain')['visitNumber'].transform('mean')
    data_df['min_visit_per_network_domain']= data_df.groupby('geoNetwork_networkDomain')['visitNumber'].transform('min')
    data_df['max_visit_per_network_domain']= data_df.groupby('geoNetwork_networkDomain')['visitNumber'].transform('max')
    data_df['sum_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('sum')
    data_df['count_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('count')
    data_df['mean_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('mean')
    data_df['max_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('max')
    data_df['sum_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('sum')
    data_df['count_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('count')
    data_df['mean_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('mean')
    data_df['max_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('max')
    data_df['network_domain_id_nunique'] = data_df.groupby('geoNetwork_networkDomain')['fullVisitorId'].transform('nunique')
    logger.info('Done with geoNetworks')
    return data_df

def process_traffic_source(data_df):
    logger.info('Start trafficSource')
    data_df['source_country'] = data_df['trafficSource_source'] + '_' + data_df['geoNetwork_country']
    data_df['campaign_medium'] = data_df['trafficSource_campaign'] + '_' + data_df['trafficSource_medium']
    data_df['campaign_country'] = data_df['trafficSource_campaign'] + '_' + data_df['geoNetwork_country']
    data_df['medium_hits_mean'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('mean')
    data_df['medium_hits_max'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('max')
    data_df['medium_hits_min'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('min')
    data_df['medium_hits_sum'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('sum')
    data_df['medium_hits_var'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('var')
    logger.info('Done with trafficSource')
    return data_df

def drop_convert_columns(train_df=None, test_df=None):
    logger.info('Start drop convert')
    cols_to_drop = [col for col in train_df.columns if train_df[col].nunique(dropna=False) == 1]
    train_df.drop(cols_to_drop, axis=1, inplace=True)
    test_df.drop([col for col in cols_to_drop if col in test_df.columns], axis=1, inplace=True)
    ###only one not null value
    train_df.drop(['trafficSource_campaignCode'], axis=1, inplace=True)
    ###converting columns format
    train_df['totals_transactionRevenue'] = train_df['totals_transactionRevenue'].astype(float)
    train_df['totals_transactionRevenue'] = train_df['totals_transactionRevenue'].fillna(0)
    train_df['totals_transactionRevenue'] = np.log1p(train_df['totals_transactionRevenue'])
    logger.info('Done with  drop convert')
    return train_df, test_df
    
    
def process_categorical_columns(train_df=None, test_df=None):
    ## Categorical columns
    logger.info('Process categorical columns ...')
    num_cols = ['month_unique_user_count', 'day_unique_user_count', 'weekday_unique_user_count', 'weekofyear_unique_user_count',
                'visits_id_browser_mean', 'visits_id_os_mean','visits_id_cat_mean',
                'visitNumber',  'visits_id_sum', 'visits_id_min', 'visits_id_max', 'visits_id_mean', 'visits_id_nunique',
                'browser_unique_user_count', 'os_unique_user_count',
                'totals_hits', 'hits_id_sum', 'hits_id_cnt',
                'totals_pageviews', 'pageviews_id_sum', 'pageviews_id_cnt',
                'mean_visit_per_network_domain','min_visit_per_network_domain','max_visit_per_network_domain',
                'mean_hits_per_day', 'sum_hits_per_day', 'min_hits_per_day', 'max_hits_per_day', 'var_hits_per_day',
                'mean_pageviews_per_day', 'sum_pageviews_per_day', 'min_pageviews_per_day', 'max_pageviews_per_day',
                'sum_pageviews_per_network_domain', 'count_pageviews_per_network_domain', 'mean_pageviews_per_network_domain','max_pageviews_per_network_domain',
                'sum_hits_per_network_domain', 'count_hits_per_network_domain', 'mean_hits_per_network_domain', 'max_hits_per_network_domain', 'network_domain_id_nunique',
                'medium_hits_mean','medium_hits_min','medium_hits_max','medium_hits_sum', 'medium_hits_var']
                
    not_used_cols = ["date", "fullVisitorId", "sessionId", 
            "visitId", "visitStartTime", 'totals_transactionRevenue', 'trafficSource_referralPath']
    cat_cols = [col for col in train_df.columns if col not in num_cols and col not in not_used_cols]
    for col in cat_cols:
        logger.info('Process categorical columns:{}'.format(col))
        lbl = LabelEncoder()
        lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
        train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
        test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))
    return not_used_cols, train_df, test_df

def model(train_df=None, test_df=None,not_used_cols=None):
    logger.info('Start prepare model')
    train_df = train_df.sort_values('date')
    X_test = test_df.drop([col for col in not_used_cols if col in test_df.columns], axis=1)
    
    ## Model parameters
    params = {
        "objective" : "regression",
        "metric" : "rmse", 
        "num_leaves" : 30,
        "min_child_samples" : 100,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 20180926,
        "verbosity" : -1
    }
   
    ## Model
    logger.info('Start traininig model')
    model = lgb.LGBMRegressor(**params, n_estimators = NUM_ROUNDS, nthread = 4, n_jobs = -1)
    
    prediction = np.zeros(test_df.shape[0])
    
    if(TIME_SPLIT):
        import datetime
        train = train_df[train_df['date']<=datetime.date(2017,6,15)]
        valid = train_df[train_df['date']>datetime.date(2017,6,15)]
        X_train = train.drop(not_used_cols, axis=1)
        y_train = train['totals_transactionRevenue']
        X_valid = valid.drop(not_used_cols, axis=1)
        y_valid = valid['totals_transactionRevenue']
        model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',
                    verbose=VERBOSE_EVAL, early_stopping_rounds=STOP_ROUNDS)
        prediction = model.predict(X_test, num_iteration=model.best_iteration_)           
    else:
        X = train_df.drop(not_used_cols, axis=1)
        y = train_df['totals_transactionRevenue']
        folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=20180917)
        for fold_n, (train_index, test_index) in enumerate(folds.split(X)):
            print(fold_n)
            X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',
                    verbose=VERBOSE_EVAL, early_stopping_rounds=STOP_ROUNDS)
            
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            prediction += y_pred
            logger.info('Done with Fold:{}'.format(fold_n))
        prediction /= N_SPLITS
    
    return prediction

def submission(test_df=None, prediction=None):
    # Submission
    logger.info('Prepare submission')
    submission = test_df[['fullVisitorId']].copy()
    submission.loc[:, 'PredictedLogRevenue'] = prediction
    submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
    submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0.0)
    submission["PredictedLogRevenue"] = np.expm1(prediction)
    submission = submission.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
    submission.columns = ["fullVisitorId", "PredictedLogRevenue"]
    submission["PredictedLogRevenue"] = np.log1p(submission["PredictedLogRevenue"])
    submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0.0)
    submission.to_csv("submission_20180926.csv", index=False)
    logger.info('Done with submission')

    
def main(sum_of_logs=False, nrows=None):
    #Feature processing
    ## Load data

    train_df = read_parse_dataframe('train_v2.csv',nrows=nrows)
    train_df = process_date_time(train_df)
    test_df = read_parse_dataframe('test_v2.csv',nrows=nrows)
    test_df = process_date_time(test_df)
    
    ## Drop columns
    train_df, test_df = drop_convert_columns(train_df, test_df)
    
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
    not_used_cols, train_df, test_df = process_categorical_columns(train_df, test_df)
    
    # Model
    prediction = model(train_df, test_df, not_used_cols)
    
    #Submission
    submission(test_df, prediction)
    
 
if __name__ == "__main__":
    main(nrows=100000)
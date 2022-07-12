### I set some ideas together ###### I used a lot of scripts and models of kaggle to construct this

### It's a simple model.


import numpy as np
import pandas as pd
import gc
import time
import os
from contextlib import contextmanager
from lightgbm import LGBMRegressor
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
from datetime import datetime
import json # to convert json in df
from pandas.io.json import json_normalize # to normalize the json file
from sklearn.preprocessing import LabelEncoder


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    
def json_read(df, percentualOfTotal = None):
    
    data_frame = "../input/" + df
    
    columns = ['device', 'geoNetwork', 'totals', 'trafficSource'] # Columns that have json format

    #Importing the dataset
    df = pd.read_csv(data_frame, 
                     converters={column: json.loads for column in columns}, # loading the json columns properly
                     dtype={'fullVisitorId': 'str'}, # transforming Id to string
                     skiprows=lambda i: i>0 and random.random() > percentualOfTotal)# Number of rows that will be imported randomly
    
    for column in columns: #loop to finally transform the columns in data frame
        #It will normalize and set the json to a table
        column_as_df = json_normalize(df[column]) 
        # here will be set the name using the category and subcategory of json columns
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns] 
        # after extracting the values, let drop the original columns
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        
    
    return df # returning the df after importing and transforming


def loadingData(p):
    # Read data and merge
    df = json_read('train.csv', percentualOfTotal = p)
    df_test = json_read('test.csv', percentualOfTotal = p)
    print("Train samples: {}, train fullVisitorId uniques: {}".format(len(df), len(df['fullVisitorId'])))
    print("Test samples: {}, test fullVisitorId uniques: {}".format(len(df_test), len(df_test['fullVisitorId'])))

    #filling the target in trainning df
    df["totals.transactionRevenue"] = df["totals.transactionRevenue"].fillna(0.0).astype(float) #filling NA with zero  
    df['totals.transactionRevenue'] = df['totals.transactionRevenue'].apply(lambda x: np.log1p(x))
    
    df = df.append(df_test, sort=False).reset_index()
    
    df['_buyCount'] = df.groupby('fullVisitorId').cumcount() + 1
    
    print("Total DataFrame shape: {}, Total fullVisitorId uniques: {}".format(df.shape, len(df['fullVisitorId'])))
    
    not_aval_cols = ['socialEngagementType','device.browserSize','device.browserVersion', 'device.flashVersion', 'sessionId',
                     'device.language' ,'device.mobileDeviceBranding', 'device.mobileDeviceInfo','device.mobileDeviceMarketingName',
                     'device.mobileDeviceModel', 'device.mobileInputSelector', 'device.operatingSystemVersion','device.screenColors',
                     'device.screenResolution', 'geoNetwork.cityId', 'geoNetwork.latitude' ,'geoNetwork.longitude',
                     'geoNetwork.networkLocation','trafficSource.adwordsClickInfo.criteriaParameters', 'visitId','index']
    
    df.drop(not_aval_cols, axis=1, inplace=True)

    df['dummy'] = 1
    df['user_cumcnt_per_day'] = (df[['fullVisitorId','date', 'dummy']].groupby(['fullVisitorId','date'])['dummy'].cumcount()+1)
    df['user_sum_per_day'] = df[['fullVisitorId','date', 'dummy']].groupby(['fullVisitorId','date'])['dummy'].transform(sum)
    df['user_cumcnt_sum_ratio_per_day'] = df['user_cumcnt_per_day'] / df['user_sum_per_day'] 
    
    df.drop('dummy', axis=1, inplace=True)

    del df_test
    
    return df

def date_process(df):
    df["date"] = pd.to_datetime(df['visitStartTime'], unit='s') # seting the column as pandas datetime
    df["_weekday"] = df['date'].dt.weekday #extracting week day
    df["_day"] = df['date'].dt.day # extracting day
    df["_month"] = df['date'].dt.month # extracting day
    df["_year"] = df['date'].dt.year # extracting day
    df['_visitHour'] = (df['visitStartTime'].apply(lambda x: str(datetime.fromtimestamp(x).hour))).astype(int)
    df.sort_values(['fullVisitorId', 'date'], ascending=True, inplace=True)
    del df['date']
    del df['visitStartTime']
    
    return df #returning the df after the transformations


def FillingNaValues(df):    # fillna numeric feature
    df['totals.pageviews'].fillna(1, inplace=True) #filling NA's with 1
    df['totals.newVisits'].fillna(0, inplace=True) #filling NA's with 0
    df['totals.bounces'].fillna(0, inplace=True)   #filling NA's with 0
    df['trafficSource.isTrueDirect'].fillna(False, inplace=True) # filling boolean with False
    df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True, inplace=True) # filling boolean with True
    df.loc[df['geoNetwork.city'] == "(not set)", 'geoNetwork.city'] = np.nan
    df['geoNetwork.city'].fillna("notInform", inplace=True)
    df['totals.hits'] = df['totals.hits'].astype(int)
    df['totals.pageviews'] = df['totals.pageviews'].astype(int)
    df['totals.newVisits'] = df['totals.newVisits'].astype(int)
    df['totals.bounces'] = df['totals.bounces'].astype(int)
    df['trafficSource.isTrueDirect'] = df['trafficSource.isTrueDirect'].astype(bool)
    df['trafficSource.adwordsClickInfo.isVideoAd'] = df['trafficSource.adwordsClickInfo.isVideoAd'].astype(bool)
    
    return df #return the transformed dataframe

def processTotalsFeatures(df):
    df['visits_id_sum'] = df.groupby(['fullVisitorId'])['visitNumber'].transform('sum')
    df['visits_id_min'] =  df.groupby(['fullVisitorId'])['visitNumber'].transform('min')
    df['visits_id_max'] = df.groupby(['fullVisitorId'])['visitNumber'].transform('max')
    df['visits_id_mean'] = df.groupby(['fullVisitorId'])['visitNumber'].transform('mean')
    
    df['visits_id_nunique'] = df.groupby('visitNumber')['fullVisitorId'].transform('nunique')
    df['hits_id_sum'] = df.groupby(['fullVisitorId'])['totals.hits'].transform('sum')
    df['hits_id_cnt'] = df.groupby(['fullVisitorId'])['totals.hits'].transform('count')
    df['pageviews_id_sum'] = df.groupby(['fullVisitorId'])['totals.pageviews'].transform('sum')
    df['pageviews_id_cnt'] = df.groupby(['fullVisitorId'])['totals.pageviews'].transform('count')
    
    df['mean_hits_per_day'] = df.groupby(['fullVisitorId','_day'])['totals.hits'].transform('mean')
    df['sum_hits_per_day'] = df.groupby(['fullVisitorId','_day'])['totals.hits'].transform('sum')
    df['max_hits_per_day'] = df.groupby(['fullVisitorId','_day'])['totals.hits'].transform('max')
    df['min_hits_per_day'] = df.groupby(['fullVisitorId','_day'])['totals.hits'].transform('min')
    
    df['mean_pageviews_per_day'] = df.groupby(['fullVisitorId','_day'])['totals.pageviews'].transform('mean')
    df['sum_pageviews_per_day'] = df.groupby(['fullVisitorId','_day'])['totals.pageviews'].transform('sum')
    df['max_pageviews_per_day'] = df.groupby(['fullVisitorId','_day'])['totals.pageviews'].transform('max')
    df['min_pageviews_per_day'] = df.groupby(['fullVisitorId','_day'])['totals.pageviews'].transform('min')   

    return df

def processGeoNetwork(df):
    df['mean_visit_per_network_domain']= df.groupby('geoNetwork.networkDomain')['visitNumber'].transform('mean')
    df['min_visit_per_network_domain']= df.groupby('geoNetwork.networkDomain')['visitNumber'].transform('min')
    df['max_visit_per_network_domain']= df.groupby('geoNetwork.networkDomain')['visitNumber'].transform('max')
    df['sum_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
    df['count_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
    df['mean_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('mean')
    df['max_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('max')
    df['sum_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
    df['count_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
    df['mean_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')
    df['max_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('max')
    df['network_domain_id_nunique'] = df.groupby('geoNetwork.networkDomain')['fullVisitorId'].transform('nunique')
    
    return df

def processTrafficSource(df):
    df['source_country'] = df['trafficSource.source'] + '_' + df['geoNetwork.country']
    df['campaign_medium'] = df['trafficSource.campaign'] + '_' + df['trafficSource.medium']
    df['campaign_country'] = df['trafficSource.campaign'] + '_' + df['geoNetwork.country']
    df['medium_hits_mean'] = df.groupby(['trafficSource.medium'])['totals.hits'].transform('mean')
    df['medium_hits_max'] = df.groupby(['trafficSource.medium'])['totals.hits'].transform('max')
    df['medium_hits_min'] = df.groupby(['trafficSource.medium'])['totals.hits'].transform('min')
    df['medium_hits_sum'] = df.groupby(['trafficSource.medium'])['totals.hits'].transform('sum')
    df['medium_hits_var'] = df.groupby(['trafficSource.medium'])['totals.hits'].transform('var')

    return df

def FeatureEngineering(df):
    
    df['visitNumber'] = np.log1p(df['visitNumber'])

    df['totals.hits'] = np.log1p(df['totals.hits'])

    df['totals.pageviews'] = np.log1p(df['totals.pageviews'])

    df['sum_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
    df['count_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
    df['mean_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('mean')
    df['sum_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
    df['count_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
    df['mean_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')

    df['mean_hits_per_day'] = df.groupby(['_day'])['totals.hits'].transform('mean')
    df['sum_hits_per_day'] = df.groupby(['_day'])['totals.hits'].transform('sum')

    df['user_mean_hits_per_hour'] = df.groupby(['fullVisitorId','_visitHour'])['totals.hits'].transform('mean')
    df['user_sum_hits_per_hour'] = df.groupby(['fullVisitorId','_visitHour'])['totals.hits'].transform('sum')
    df['user_mean_pageviews_per_browser'] = df.groupby(['fullVisitorId','device.browser'])['totals.pageviews'].transform('mean')
    df['user_sum_pageviews_per_browser'] = df.groupby(['fullVisitorId','device.browser'])['totals.pageviews'].transform('sum')
    df['user_mean_pageviews_per_CGroup'] = df.groupby(['fullVisitorId','channelGrouping'])['totals.pageviews'].transform('mean')
    df['user_sum_pageviews_per_CGroup'] = df.groupby(['fullVisitorId','channelGrouping'])['totals.pageviews'].transform('sum')
        
    df['user_count_pageviews_per_domain'] = df.groupby(['fullVisitorId','geoNetwork.networkDomain'])['totals.pageviews'].transform('count')
    df['user_sum_pageviews_per_domain'] = df.groupby(['fullVisitorId','geoNetwork.networkDomain'])['totals.pageviews'].transform('sum')    
    
    df['user_count_pageviews_per_source'] = df.groupby(['fullVisitorId','trafficSource.source'])['totals.pageviews'].transform('count')
    df['user_sum_pageviews_per_source'] = df.groupby(['fullVisitorId','trafficSource.source'])['totals.pageviews'].transform('sum')        
    df['user_count_pageviews_per_source'] = df.groupby(['fullVisitorId','trafficSource.source'])['totals.hits'].transform('count')
    df['user_sum_pageviews_per_source'] = df.groupby(['fullVisitorId','trafficSource.source'])['totals.hits'].transform('sum')        
                  
    df['user_count_pageviews_per_hour'] = df.groupby(['fullVisitorId','_visitHour'])['totals.pageviews'].transform('count')
    df['user_sum_pageviews_per_hour'] = df.groupby(['fullVisitorId','_visitHour'])['totals.pageviews'].transform('sum')
        
    df['sum_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
    df['count_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
    df['mean_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('mean')

    df['sum_pageviews_per_region'] = df.groupby('geoNetwork.region')['totals.pageviews'].transform('sum')
    df['count_pageviews_per_region'] = df.groupby('geoNetwork.region')['totals.pageviews'].transform('count')
    df['mean_pageviews_per_region'] = df.groupby('geoNetwork.region')['totals.pageviews'].transform('mean')

    df['sum_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
    df['count_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
    df['mean_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')

    df['sum_hits_per_region'] = df.groupby('geoNetwork.region')['totals.hits'].transform('sum')
    df['count_hits_per_region'] = df.groupby('geoNetwork.region')['totals.hits'].transform('count')
    df['mean_hits_per_region'] = df.groupby('geoNetwork.region')['totals.hits'].transform('mean')

    df['sum_hits_per_country'] = df.groupby('geoNetwork.country')['totals.hits'].transform('sum')
    df['count_hits_per_country'] = df.groupby('geoNetwork.country')['totals.hits'].transform('count')
    df['mean_hits_per_country'] = df.groupby('geoNetwork.country')['totals.hits'].transform('mean')

    df['user_pageviews_sum'] = df.groupby('fullVisitorId')['totals.pageviews'].transform('sum')
    df['user_hits_sum'] = df.groupby('fullVisitorId')['totals.hits'].transform('sum')


    df['user_pageviews_count'] = df.groupby('fullVisitorId')['totals.pageviews'].transform('count')
    df['user_hits_count'] = df.groupby('fullVisitorId')['totals.hits'].transform('count')

    df['user_pageviews_sum_to_mean'] = df['user_pageviews_sum'] / df['user_pageviews_sum'].mean()
    df['user_hits_sum_to_mean'] = df['user_hits_sum'] / df['user_hits_sum'].mean()
    
    df['user_pageviews_to_region'] = df['user_pageviews_sum'] / df['mean_pageviews_per_region']
    df['user_hits_to_region'] = df['user_hits_sum'] / df['mean_hits_per_region']
    
    return df


def featurePreprocessing(df):
    
    numerical_cols = df.select_dtypes(include=float)
    
    no_use = ["visitNumber", "date", "fullVisitorId", "sessionId", 
              "visitId", "visitStartTime", 'totals.transactionRevenue']

    cat_cols = [col for col in df.columns if col not in numerical_cols and col not in no_use]
    
    for col in cat_cols:
        #print(col)
        lbl = LabelEncoder()
        lbl.fit(list(df[col].values.astype('str')) + list(df[col].values.astype('str')))
        df[col] = lbl.transform(list(df[col].values.astype('str')))
        
    for col in ['trafficSource.keyword', 'trafficSource.referralPath', 'trafficSource.adwordsClickInfo.gclId',
                'trafficSource.adwordsClickInfo.adNetworkType',  'trafficSource.adwordsClickInfo.isVideoAd',
                'trafficSource.adwordsClickInfo.page', 'trafficSource.adwordsClickInfo.slot', 'trafficSource.adContent']:
        df[col].fillna('unknown', inplace=True)


        
    return df

def kfold_lightgbm(df, num_folds, stratified = False, debug= False):

    # Divide in training/validation and test data
    train_df = df[df['totals.transactionRevenue'].notnull()]
    test_df = df[df['totals.transactionRevenue'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    
    train_id = train_df['fullVisitorId']
    test_id = test_df['fullVisitorId']
    
    train_df.drop('fullVisitorId', axis=1, inplace=True)
    test_df.drop('fullVisitorId', axis=1, inplace=True)
    
    del df
    gc.collect()
    
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['totals.transactionRevenue',"fullVisitorId"]]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['totals.transactionRevenue'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['totals.transactionRevenue'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['totals.transactionRevenue'].iloc[valid_idx]

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

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMRegressor(**params, nthread=4, 
                            n_estimators=10000)

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric="rmse",
                verbose= 1, early_stopping_rounds= 10000)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()


    # Write submission file and plot feature importance
    if not debug:
        submission_google = "google_pred.csv"
        test_df['fullVisitorId'] = test_id
        test_df['totals.transactionRevenue'] = sub_preds
        test_df[['fullVisitorId', 'totals.transactionRevenue']].to_csv(submission_google, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df
 
# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')
     


def main(debug = False):
    p = 0.01 if debug else 1
    df = loadingData(p)

    with timer("Filling Missing Values: "):
        df = FillingNaValues(df)
        print("Shape after Filling Missing", df.shape)
        gc.collect();

    with timer("Dates Features Processing: "):
        df = date_process(df)
        print("Data Shape after Processing: ", df.shape)
        gc.collect();

    with timer("Totals Features Processing: "):
        df = processTotalsFeatures(df)
        print("Data Shape after Processing: ", df.shape)    
        gc.collect();

    with timer("Geonetwork Features Processing: "):
        df = processGeoNetwork(df)
        print("Shape after Geonetwork: ", df.shape)
        gc.collect()

    with timer("Feature Engineering: "):
        df = FeatureEngineering(df)
        print("Shape After Feature Engineering: ", df.shape)
        gc.collect()
        
    with timer("Feature Preprocessing: "):
        df = featurePreprocessing(df)
        print("Shape After Preprocessing: ", df.shape)
        gc.collect()

    with timer("Run LightGBM with kfold"):

        feat_importance = kfold_lightgbm(df, num_folds= 2, stratified= False, debug= debug)


if __name__ == "__main__":
    with timer("Full model run"):
        df = main()
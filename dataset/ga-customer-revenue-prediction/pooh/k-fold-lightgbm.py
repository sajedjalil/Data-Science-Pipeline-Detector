#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 07:57:15 2018

@author: ashunigion

"""


#important imports 

import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from pathlib import Path
import glob
import matplotlib.pyplot as plt
from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
from plotly.offline import iplot, init_notebook_mode
from sklearn import model_selection, preprocessing, metrics
import datetime
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import gc


from sklearn.model_selection import KFold, StratifiedKFold
#declaring global variables
csvPath = "../input/train-dev.csv"
#csvTestPath
nrowDev = 100000
nrowTrain = None
dev = False
labPath = "./labSpace/"

#function to load the data in the json format and then flatten them to the CSV format
#author of this function[https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields/notebook]
def flattenJSON(csv_Path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_Path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_Path)}. Shape: {df.shape}")
    return df


#creating the temporary space for experimenting
def labSpace(path = labPath):
   if Path(path).exists():
       files = glob.glob(path+"*")
       for f in files:
           os.remove(f)
       Path(path).rmdir()
   Path(labPath).mkdir(mode=0o777, parents=False, exist_ok=False)
   
#fillZero in place of the NA values for the transaction values
def fillZero(df):
    df["totals.transactionRevenue"].fillna(0, inplace=True)


#function to encode the categorical labels
def catEncode(train_df,test_df):
    cat_cols = ["channelGrouping", "device.browser", 
            "device.deviceCategory", "device.operatingSystem", 
            "geoNetwork.city", "geoNetwork.continent", 
            "geoNetwork.country", "geoNetwork.metro",
            "geoNetwork.networkDomain", "geoNetwork.region", 
            "geoNetwork.subContinent", "trafficSource.adContent", 
            "trafficSource.adwordsClickInfo.adNetworkType", 
            "trafficSource.adwordsClickInfo.gclId", 
            "trafficSource.adwordsClickInfo.page", 
            "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
            "trafficSource.keyword", "trafficSource.medium", 
            "trafficSource.referralPath", "trafficSource.source",
            'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.isTrueDirect']
    for col in cat_cols:
        print(col)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
        train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
        test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))
    return cat_cols
        
#function to convert the numerical variables to floats
def toFloat(train_df,test_df):
    num_cols = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',  'totals.newVisits']    
    for col in num_cols:
        train_df[col] = train_df[col].astype(float)
        test_df[col] = test_df[col].astype(float)
    return num_cols




#conversion of date to proper format
def conv(df):
    df['date'] = df['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
    df["totals.transactionRevenue"] = df["totals.transactionRevenue"].astype('float')

       
# custom function to run light gbm model
def kfold_lightgbm(df_x,df_y,test, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df_x
    test_df = test
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df_x
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns ]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], df_y)):
        train_x, train_y = train_df[feats].iloc[train_idx], df_y[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], df_y[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMRegressor(
            objective = "regression",
            metric = "rmse", 
            num_leaves = 30,
            min_child_samples = 100,
            learning_rate = 0.1,
            bagging_fraction = 0.7,
            feature_fraction = 0.5,
            bagging_frequency = 5,
            bagging_seed = 2018,
            verbosity = -1 )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'rmse', verbose= 100, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration_)
        sub_preds += clf.predict(test_df[feats], num_iteration=clf.best_iteration_) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, mean_squared_error(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full RMSE score %.6f' % mean_squared_error(df_y, oof_preds))
    # Write submission file and plot feature importance
#    if not debug:
#        test_df['TARGET'] = sub_preds
#        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
#    display_importances(feature_importance_df)
    return sub_preds

#create submission
def createSub(test_id,pred_test):
    sub_df = pd.DataFrame({"fullVisitorId":test_id})
    pred_test[pred_test<0] = 0
    sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
    sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
    sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
    sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
    sub_df.to_csv("KFold_lgb.csv", index=False)
    
#main function call
if __name__ == '__main__':
    
    labSpace()
    if(dev):
        df_train = flattenJSON('../input/train.csv',nrowDev)
        df_test = flattenJSON('../input/test.csv',nrowDev)
    else:
        df_train = flattenJSON('../input/train.csv',nrowTrain)
        df_test = flattenJSON('../input/test.csv',nrowTrain)
    #convert the date in proper format
    conv(df_train)
    #the code for visualization
    #explore(df_train)
    #code to train the model and preprocess the data
    
    print("Variables not in test but in train : ", set(df_train.columns).difference(set(df_test.columns)))
    
    
    const_cols = [c for c in df_train.columns if df_train[c].nunique(dropna=False)==1 ]
    cols_to_drop = const_cols + ['sessionId']

    df_train = df_train.drop(cols_to_drop + ["trafficSource.campaignCode"], axis=1)
    df_test = df_test.drop(cols_to_drop, axis=1)
    fillZero(df_train)
    
    train_y = df_train["totals.transactionRevenue"].values
    train_id = df_train["fullVisitorId"].values
    test_id = df_test["fullVisitorId"].values
    
    #function to encode the categorical variables
    cat_cols = catEncode(df_train,df_test)
    #changing the numeric type to the float
    num_cols = toFloat(df_train,df_test)
    
    #the spilt for the purpose of training and model development
    dev_df = df_train
    dev_y = np.log1p(dev_df["totals.transactionRevenue"].values) #calculates log(1+x)
    
    
    dev_X = dev_df[cat_cols + num_cols] 
    test_X = df_test[cat_cols + num_cols] 
    
    
    # Training the model #
    pred_test = kfold_lightgbm(dev_X,dev_y,test_X, 5)
    
    #create submission
    createSub(test_id,pred_test)
    
    
    print("success")


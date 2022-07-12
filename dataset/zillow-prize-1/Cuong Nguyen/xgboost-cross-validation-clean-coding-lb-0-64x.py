# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
from datetime import datetime
import gc

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


## STEP0: do setting
class Settings(Enum):
    global train_path
    global properties_path
    global submission_path
        
    train_path      = '../input/train_2016_v2.csv'
    properties_path = '../input/properties_2016.csv'
    submission_path = "../input/sample_submission.csv"
        
    def __str__(self):
        return self.value
        
    
## STEP1: process data    
def fill_NA(df):
    print('\nFilling NA ...')
    
    na_ratio = ((df.isnull().sum() / len(df)) * 100).sort_values(ascending=False)
    print('NA ratio: ')
    print(na_ratio) 
    
    df['hashottuborspa'] = df['hashottuborspa'].fillna("FALSE")
    df['fireplaceflag'] = df['fireplaceflag'].fillna("FALSE")
    
    for feature in df:
        if df[feature].dtype == 'object':
            df[feature] = df[feature].fillna("None")
        else:
            df[feature] = df[feature].fillna(0)
    
def encode_features(df):
    print('\nEncoding features ...')
    
    for feature in df:
        if df[feature].dtype == 'object':
            print('Encoding ', feature)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[feature].values))
            df[feature] = lbl.transform(list(df[feature].values))
    
def display_feature_target(df, feature):
    fig, ax = plt.subplots()
    ax.scatter(x = df[feature], y = df['logerror'])
    plt.ylabel('logerror', fontsize=13)
    plt.xlabel(feature, fontsize=13)
    plt.show()
    
def _process_data():
    print('\n\nSTEP1: processing data ...')
    
    global train_x
    global train_y
    global valid_x
    global valid_y
    global test_x
        
    # load data
    print('\nLoading data ...')
    train_df = pd.read_csv(train_path, parse_dates=["transactiondate"])
    prop_df = pd.read_csv(properties_path)

    # fill NA
    fill_NA(prop_df)
    
    # encode features
    encode_features(prop_df)
    
    # add features
    zip_count = prop_df['regionidzip'].value_counts().to_dict()
    city_count = prop_df['regionidcity'].value_counts().to_dict()
    prop_df['N-zip_count'] = prop_df['regionidzip'].map(zip_count)
    prop_df['N-city_count'] = prop_df['regionidcity'].map(city_count)
    prop_df['N-GarPoolAC'] = ((prop_df['garagecarcnt']>0) & 
         (prop_df['pooltypeid10']>0) & (prop_df['airconditioningtypeid']!=5))*1
    
    # prepare train and valid data
    print('\nPreparing train and valid data ...')
    
    drop_vars = ['parcelid', 'airconditioningtypeid', 'buildingclasstypeid',
        'buildingqualitytypeid', 'regionidcity', 'transactiondate']
    
    train_x = train_df.merge(prop_df, how='left', on='parcelid')
    
    train_x = train_x[ train_x.logerror > -0.4 ]
    train_x = train_x[ train_x.logerror < 0.419 ]

    train_x.drop(drop_vars, axis=1, inplace=True)
    
    train_y = train_x['logerror']
    train_x.drop(['logerror'], axis=1, inplace=True)

    select_qtr4 = pd.to_datetime(train_df["transactiondate"]).dt.month > 9
    train_x, valid_x = train_x[~select_qtr4], train_x[select_qtr4]
    train_y, valid_y = train_y[~select_qtr4], train_y[select_qtr4]
    
    print('train x shape: ', train_x.shape)
    print('train y shape: ', train_y.shape)
    print('valid x shape: ', valid_x.shape)
    print('valid y shape: ', valid_y.shape)
    
    # prepare test data
    print('\nPreparing test data ...')
    
    test_x = prop_df[train_x.columns]
    print('test x shape: ', test_x.shape)
    
    # release
    del train_df
    del prop_df
    gc.collect()
    
    
## STEP2: build model
def _build_model():
    print('\n\nSTEP2: building model ...')
    
    global xgb_params
    xgb_params = {
        'eta': 0.007,
        'max_depth': 6, 
        'subsample': 0.6,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'lambda': 9.0,
        'alpha': 0.8,
        'colsample_bytree': 0.7,
        'silent': 1
    }
    

## STEP3: train    
def _train():
    print('\n\nSTEP3: training ...')
    
    global xgb_clf
    
    d_train = xgb.DMatrix(train_x, label=train_y)
    d_valid = xgb.DMatrix(valid_x, label=valid_y)
    
    evals = [(d_train, 'train'), (d_valid, 'valid')]
    xgb_clf = xgb.train(xgb_params, d_train, num_boost_round=10000, evals=evals, 
                        early_stopping_rounds=100, verbose_eval=10)
    
    
## STEP4: predict
def _predict():
    print('\n\nSTEP4: predicting ...')
    
    global pred
    
    d_test = xgb.DMatrix(test_x)
    pred = xgb_clf.predict(d_test)
    
    
## STEP5: generate submission    
def _generate_submission():
    print('\n\nSTEP5: generating submission ...')

    submission = pd.read_csv(submission_path)
    for c in submission.columns[submission.columns != 'ParcelId']:
        submission[c] = pred
        
    submission.to_csv('sub{}.csv'.format(datetime.now().\
                      strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.5f')


## main
def main():
    _process_data()
    _build_model()
    _train()
    _predict()
    _generate_submission()
    

if __name__ == "__main__":
    main()
    print('\n\n\nThe end.')
    
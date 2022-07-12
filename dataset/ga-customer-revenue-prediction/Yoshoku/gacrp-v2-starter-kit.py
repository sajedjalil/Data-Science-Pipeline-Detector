# References:
# https://www.kaggle.com/ashishpatel26/1-64-plb-feature-engineering-best-model-combined
# https://www.kaggle.com/fabiendaniel/lgbm-rf-starter-lb-1-70
# https://www.kaggle.com/gpreda/ga-customer-revenue-simple-lightgbm
# https://www.kaggle.com/codlife/pre-processing-for-huge-train-data-with-chunksize

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import time
from datetime import datetime
from pandas.io.json import json_normalize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm
import matplotlib.pyplot as plt
import seaborn as sns
import gc
gc.enable()

def load_df(csv_path, chunksize=100000):
    features = ['channelGrouping', 'date', 'fullVisitorId', 'visitId',
                'visitNumber', 'visitStartTime', 'device_browser',
                'device_deviceCategory', 'device_isMobile', 'device_operatingSystem',
                'geoNetwork_city', 'geoNetwork_continent', 'geoNetwork_country',
                'geoNetwork_metro', 'geoNetwork_networkDomain', 'geoNetwork_region',
                'geoNetwork_subContinent', 'totals_bounces', 'totals_hits',
                'totals_newVisits', 'totals_pageviews', 'totals_transactionRevenue',
                'trafficSource_adContent', 'trafficSource_campaign',
                'trafficSource_isTrueDirect', 'trafficSource_keyword',
                'trafficSource_medium', 'trafficSource_referralPath',
                'trafficSource_source']
    JSON_COLS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    print('Load {}'.format(csv_path))
    df_reader = pd.read_csv(csv_path,
                            converters={ column: json.loads for column in JSON_COLS },
                            dtype={ 'date': str, 'fullVisitorId': str, 'sessionId': str },
                            chunksize=chunksize)
    res = pd.DataFrame()
    for cidx, df in enumerate(df_reader):
        df.reset_index(drop=True, inplace=True)
        for col in JSON_COLS:
            col_as_df = json_normalize(df[col])
            col_as_df.columns = ['{}_{}'.format(col, subcol) for subcol in col_as_df.columns]
            df = df.drop(col, axis=1).merge(col_as_df, right_index=True, left_index=True)
        res = pd.concat([res, df[features]], axis=0).reset_index(drop=True)
        del df
        gc.collect()
        print('{}: {}'.format(cidx + 1, res.shape))
    return res
  
def process_date_time(df):
    print('process date')
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    return df

def process_format(df):
    print('process format')
    for col in ['visitNumber', 'totals_hits', 'totals_pageviews']:
        df[col] = df[col].astype(float)
    df['trafficSource_isTrueDirect'].fillna(False, inplace=True)
    return df

def process_device(df):
    print('process device')
    df['browser_category'] = df['device_browser'] + '_' + df['device_deviceCategory']
    df['browser_operatingSystem'] = df['device_browser'] + '_' + df['device_operatingSystem']
    df['source_country'] = df['trafficSource_source'] + '_' + df['geoNetwork_country']
    return df

def process_geo_network(df):
    print('process geo network')
    df['count_hits_nw_domain'] = df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('count')
    df['sum_hits_nw_domain'] = df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('sum')
    df['count_pvs_nw_domain'] = df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('count')
    df['sum_pvs_nw_domain'] = df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('sum')
    return df

def process_categorical_cols(train_df, test_df, excluded_cols):
    # Label encoding
    objt_cols = [col for col in train_df.columns if col not in excluded_cols and train_df[col].dtypes == object]
    for col in objt_cols:
        train_df[col], indexer = pd.factorize(train_df[col])
        test_df[col] = indexer.get_indexer(test_df[col])
    bool_cols = [col for col in train_df.columns if col not in excluded_cols and train_df[col].dtypes == bool]
    for col in bool_cols:
        train_df[col] = train_df[col].astype(int)
        test_df[col] = test_df[col].astype(int)
    # Fill NaN
    numb_cols = [col for col in train_df.columns if col not in excluded_cols and col not in objt_cols]
    for col in numb_cols:
        train_df[col] = train_df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)
    return train_df, test_df

def process_dfs(train_df, test_df, target_values, excluded_cols):
    print('Dropping repeated columns')
    cols_to_drop = [col for col in train_df.columns if train_df[col].nunique(dropna=False) == 1]
    train_df.drop(cols_to_drop, axis=1, inplace=True)
    test_df.drop([col for col in cols_to_drop if col in test_df.columns], axis=1, inplace=True)
    print('Extracting features')
    print('Training set:')
    train_df = process_date_time(train_df)
    train_df = process_format(train_df)
    train_df = process_device(train_df)
    train_df = process_geo_network(train_df)
    print('Testing set:')
    test_df = process_date_time(test_df)
    test_df = process_format(test_df)
    test_df = process_device(test_df)
    test_df = process_geo_network(test_df)
    print('Postprocess')
    train_df, test_df = process_categorical_cols(train_df, test_df, excluded_cols)
    return train_df, test_df
  
def preprocess():
    # Load data set.
    train_df = load_df('../input/train_v2.csv')
    test_df = load_df('../input/test_v2.csv')
    # Obtain target values.
    target_values = np.log1p(train_df['totals_transactionRevenue'].fillna(0).astype(float))
    # Extract features.
    EXCLUDED_COLS = ['date', 'fullVisitorId', 'visitId', 'visitStartTime', 'totals_transactionRevenue']
    train_df, test_df = process_dfs(train_df, test_df, target_values, EXCLUDED_COLS)
    test_fvid = test_df[['fullVisitorId']].copy()
    train_df.drop(EXCLUDED_COLS, axis=1, inplace=True)
    test_df.drop(EXCLUDED_COLS, axis=1, inplace=True)
    return train_df, target_values, test_df, test_fvid
    
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
  
def submit(predictions, submission, filename):
    submission.loc[:, 'PredictedLogRevenue'] = predictions
    submission['PredictedLogRevenue'] = submission['PredictedLogRevenue'].fillna(0.0).apply(lambda x : 0.0 if x < 0 else x)
    grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
    grouped_test['PredictedLogRevenue'] = np.log1p(grouped_test['PredictedLogRevenue'])
    grouped_test['PredictedLogRevenue'] = grouped_test['PredictedLogRevenue'].fillna(0.0)
    grouped_test.to_csv(filename, index=False)

def plot_importances(imps):
    mean_gain = np.log1p(imps[['gain', 'feature']].groupby('feature').mean())
    imps['mean_gain'] = imps['feature'].map(mean_gain['gain'])
    plt.figure(figsize=(8, 12))
    sns.barplot(x='gain', y='feature', data=imps.sort_values('mean_gain', ascending=False))
    plt.tight_layout()
    plt.savefig('importances.png')

def main():
    print('Start preprocessing')
    train_df, train_tvals, test_df, test_fvid = preprocess()
    feature_name = train_df.columns

    print('Start estimation')
    params = { 'metric': 'rmse' }
    est_lgbm = lightgbm.LGBMRegressor(boosting_type='gbdt',
                                      num_leaves=32, max_depth=5, learning_rate=0.01, n_estimators=10000,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                      reg_alpha=0.05, reg_lambda=0.05, random_state=1, **params)

    N_SPLITS = 5
    y_test_vec = np.zeros(test_df.shape[0])
    mean_rmse = 0.0
    importances = pd.DataFrame()
    folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=1)
    for fold_idx, (train_ids, valid_ids) in enumerate(folds.split(train_df)):
        # Split traninig data set.
        trn_x, trn_y = train_df.iloc[train_ids], train_tvals.iloc[train_ids]
        val_x, val_y = train_df.iloc[valid_ids], train_tvals.iloc[valid_ids]
        # Train estimator.
        est_lgbm.fit(trn_x, trn_y, 
                     eval_set=[(val_x, val_y)],
                     eval_metric='rmse',
                     early_stopping_rounds=50, 
                     verbose=False)
        # Prediction and evaluation on validation data set.
        val_pred = est_lgbm.predict(val_x)
        rmse_valid = rmse(val_y, np.maximum(0, val_pred))
        mean_rmse += rmse_valid
        print("%d RMSE: %.5f" % (fold_idx + 1, rmse_valid))
        # Prediction of testing data set.
        y_test_vec += np.expm1(np.maximum(0, est_lgbm.predict(test_df)))
        # Set feature importances.
        imp_df = pd.DataFrame()
        imp_df['feature'] = feature_name
        imp_df['gain'] = est_lgbm.feature_importances_
        imp_df['fold'] = fold_idx + 1
        importances = pd.concat([importances, imp_df], axis=0, sort=False)
        
    print("Mean RMSE: %.5f" % (mean_rmse / N_SPLITS))
  
    del train_df, test_df, train_tvals
    gc.collect()
    
    # Plot feature importances
    print('Plot feature importances')
    plot_importances(importances)
    
    # Save submission file.
    print('Save submission file')
    y_test_vec /= N_SPLITS
    submit(y_test_vec, test_fvid, 'submission.csv')

if __name__ == '__main__':
    main()

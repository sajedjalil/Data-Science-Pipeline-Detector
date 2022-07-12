# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import gc
import datetime as dt


## Version 3 - LB 0.0644042
# Train month averages for test predictions seem work better than their linear fit,
# so I changed it (overfitting test data as hell... but who doesn't here? ;))

## Version 2 - LB 0.0644120
# LGBM performs much better, so I left him alone

## Version 1 - LB 0.0644711
# Both models have the same weight, which is based on cross-validation results, but
# XGB model seems to be worse on public LB, 'cause alone gets score 0.0646474,
# which is much worse than score of the combination. I reached the limit of submissions,
# so I will check how LGBM alone performs tomorrow. Check it out for your own ;)


print('Loading data...')
properties = pd.read_csv('../input/properties_2016.csv', low_memory = False)
train = pd.read_csv('../input/train_2016_v2.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv', low_memory = False)
train = pd.merge(train, properties, how = 'left', on = 'parcelid')
test = pd.merge(sample_submission[['ParcelId']], properties.rename(columns = {'parcelid': 'ParcelId'}), 
                how = 'left', on = 'ParcelId')
del properties
gc.collect();


print('Memory usage reduction...')
train[['latitude', 'longitude']] /= 1e6
test[['latitude', 'longitude']] /= 1e6

train['censustractandblock'] /= 1e12
test['censustractandblock'] /= 1e12

for column in test.columns:
    if test[column].dtype == int:
        test[column] = test[column].astype(np.int32)
    if test[column].dtype == float:
        test[column] = test[column].astype(np.float32)
      
        
print('Feature engineering...')
train['month'] = pd.to_datetime(train['transactiondate']).dt.month
train = train.drop('transactiondate', axis = 1)
from sklearn.preprocessing import LabelEncoder
non_number_columns = train.dtypes[train.dtypes == object].index.values

for column in non_number_columns:
    train_test = pd.concat([train[column], test[column]], axis = 0)
    encoder = LabelEncoder().fit(train_test.astype(str))
    train[column] = encoder.transform(train[column].astype(str)).astype(np.int32)
    test[column] = encoder.transform(test[column].astype(str)).astype(np.int32)
    
feature_names = train.columns[2:]
feature_names = [feature for feature in feature_names if feature != 'month']

month_avgs = train.groupby('month').agg(['mean'])['logerror', 'mean'].values - train['logerror'].mean()

from sklearn.linear_model import LinearRegression
month_model = LinearRegression().fit(np.arange(4, 13, 1).reshape(-1, 1), 
                                     month_avgs[3:].reshape(-1, 1))
                             
                             
print('Preparing arrays and throwing out outliers...')
X_train = train[feature_names].values
y_train = train.iloc[:, 1].values
X_test = test[feature_names].values

del test
gc.collect();

month_values = train['month'].values
X_train = np.hstack([X_train, month_model.predict(month_values.reshape(-1, 1))])

X_train = X_train[np.abs(y_train) < 0.4, :]
y_train = y_train[np.abs(y_train) < 0.4]


print('Training LGBM model...')
ltrain = lgb.Dataset(X_train, label = y_train)

params = {}
params['metric'] = 'mae'
params['max_depth'] = 100
params['num_leaves'] = 32
params['feature_fraction'] = .85
params['bagging_fraction'] = .95
params['bagging_freq'] = 8
params['learning_rate'] = 0.0025
params['verbosity'] = 0

lgb_model = lgb.train(params, ltrain, verbose_eval=0, num_boost_round=2930)
                  
                  
print('Making predictions and praying for good results...')
X_test = np.hstack([X_test, np.zeros((X_test.shape[0], 1))])
folds = 20
n = int(X_test.shape[0] / folds)

for j in range(folds):
    results = pd.DataFrame()
    
    if j < folds - 1:
            X_test_ = X_test[j*n: (j+1)*n, :]
            results['ParcelId'] = sample_submission['ParcelId'].iloc[j*n: (j+1)*n]
    else:
            X_test_ = X_test[j*n: , :]
            results['ParcelId'] = sample_submission['ParcelId'].iloc[j*n: ]
            
    for month in [10, 11, 12]:
        X_test_[:, -1] = month_model.predict(month)
        assert X_test_.shape[1] == X_test.shape[1]
        y_pred = lgb_model.predict(X_test_)
        results['2016' + str(month)] = y_pred
        results['2017' + str(month)] = y_pred
        
    if j == 0:
        results_ = results.copy()
    else:
        results_ = pd.concat([results_, results], axis = 0)
    print('{}% completed'.format(round(100*(j+1)/folds)))
    
    
print('Saving predictions...')
results = results_[sample_submission.columns]
assert results.shape == sample_submission.shape
results.to_csv('submission.csv', index = False, float_format = '%.6f')
print('Done!')
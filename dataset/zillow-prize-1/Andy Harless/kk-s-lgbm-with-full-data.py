#  This is forked form Kamil Kaczmarek's Simple LGBM model (the only kernel I've found so far
#  that makes a serious attempt to combine 2016 and 2017 data while accounting for the leak),
#  but I don't think Kamil handles the leak optimally.

#  1. We should use 2017 properties data for all non-tax variables in the 2016 train set.
#     (It's more complete, and there is no known leakage outside the tax variables.)
#  2. We shoold use 2016 properties data for the tax variables in the 2016 train set.
#  3. We should use 2017 properties data for all variables in the 2017 train set.
#  4. Predictions for 2016 should use the same properties data as the 2016 train set.
#     (Otherwise they would be using tax infomration not available in 2016.)
#  5. Predictions for 2017 should use the same properties data as the 2017 train set.

# In Kamil's version, he doesn't use the 2017 tax variables for training at all.
# (They are set to NA in the 2017 train data and not merged with 2017 train data.)
# That procedure does avoid the leak, 
# but I think it also throws away potentially useful information,
# so in this verison I've tried to use all the information we can without leaking.

# (Note:  I'm not entirely comfortable with using future data to predict the past,
#   even when there isn't an identified leak, but even Kamil's version does that,
#   since it trains on the full training set that includes sales from 2017.
#   It's probably OK here, since we're only using 2016 as a sanity check.)


# Anyhow, there is a significant chance that I got the logic wrong or introduced bugs,
# so if anyone thinks this is wrong, please let me know.


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import gc
import datetime as dt



print('Loading data...')
# Load raw data
properties2016_raw = pd.read_csv('../input/properties_2016.csv', low_memory = False)
properties2017 = pd.read_csv('../input/properties_2017.csv', low_memory = False)
train2016 = pd.read_csv('../input/train_2016_v2.csv')
train2017 = pd.read_csv('../input/train_2017.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv', low_memory = False)

# Create a new version of 2016 properties data that takes all non-tax variables from 2017
taxvars = ['structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount']
tax2016 = properties2016_raw[['parcelid']+taxvars]
properties2016 = properties2017.drop(taxvars,axis=1).merge(tax2016, 
                 how='left', on='parcelid').reindex_axis(properties2017.columns, axis=1)

# Create a training data set
train2016 = pd.merge(train2016, properties2016, how = 'left', on = 'parcelid')
train2017 = pd.merge(train2017, properties2017, how = 'left', on = 'parcelid')
train = pd.concat([train2016, train2017], axis = 0)

# Create separate test data sets for 2016 and 2017
test2016 = pd.merge(sample_submission[['ParcelId']], properties2016.rename(columns = {'parcelid': 'ParcelId'}), 
                how = 'left', on = 'ParcelId')
test2017 = pd.merge(sample_submission[['ParcelId']], properties2017.rename(columns = {'parcelid': 'ParcelId'}), 
                how = 'left', on = 'ParcelId')
del properties2016, properties2017, train2016, train2017
gc.collect();


print('Memory usage reduction...')


train[['latitude', 'longitude']] /= 1e6
train['censustractandblock'] /= 1e12

def preptest(test):
    test[['latitude', 'longitude']] /= 1e6
    test[['latitude', 'longitude']] /= 1e6
    test['censustractandblock'] /= 1e12
    test['censustractandblock'] /= 1e12

    for column in test.columns:
        if test[column].dtype == int:
            test[column] = test[column].astype(np.int32)
        if test[column].dtype == float:
            test[column] = test[column].astype(np.float32)

preptest(test2016)
preptest(test2017)
        
print('Feature engineering...')
train['month'] = (pd.to_datetime(train['transactiondate']).dt.year - 2016)*12 + pd.to_datetime(train['transactiondate']).dt.month
train = train.drop('transactiondate', axis = 1)
from sklearn.preprocessing import LabelEncoder
non_number_columns = train.dtypes[train.dtypes == object].index.values

for column in non_number_columns:
    train_test = pd.concat([train[column], test2016[column], test2017[column]], axis = 0)
    encoder = LabelEncoder().fit(train_test.astype(str))
    train[column] = encoder.transform(train[column].astype(str)).astype(np.int32)
    test2016[column] = encoder.transform(test2016[column].astype(str)).astype(np.int32)
    test2017[column] = encoder.transform(test2017[column].astype(str)).astype(np.int32)
    
feature_names = [feature for feature in train.columns[2:] if feature != 'month']

month_avgs = train.groupby('month').agg('mean')['logerror'].values - train['logerror'].mean()
                             
print('Preparing arrays and throwing out outliers...')
X_train = train[feature_names].values
y_train = train['logerror'].values
X_test2016 = test2016[feature_names].values
X_test2017 = test2017[feature_names].values

del test2016, test2017;
gc.collect();

month_values = train['month'].values
month_avg_values = np.array([month_avgs[month - 1] for month in month_values]).reshape(-1, 1)
X_train = np.hstack([X_train, month_avg_values])

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

lgb_model = lgb.train(params, ltrain, valid_sets = [ltrain], verbose_eval=200, num_boost_round=2930)
                  
                  
print('Making predictions and praying for good results...')
X_test2016 = np.hstack([X_test2016, np.zeros((X_test2016.shape[0], 1))])
X_test2017 = np.hstack([X_test2016, np.zeros((X_test2017.shape[0], 1))])
folds = 20
n = int(X_test2016.shape[0] / folds)

for j in range(folds):
    results = pd.DataFrame()

    if j < folds - 1:
            X_test2016_ = X_test2016[j*n: (j+1)*n, :]
            X_test2017_ = X_test2017[j*n: (j+1)*n, :]
            results['ParcelId'] = sample_submission['ParcelId'].iloc[j*n: (j+1)*n]
    else:
            X_test2016_ = X_test2016[j*n: , :]
            X_test2017_ = X_test2017[j*n: , :]
            results['ParcelId'] = sample_submission['ParcelId'].iloc[j*n: ]

    for month in [10, 11, 12]:
        X_test2016_[:, -1] = month_avgs[month - 1]
        assert X_test2016_.shape[1] == X_test2016.shape[1]
        y_pred = lgb_model.predict(X_test2016_)
        results['2016'+ str(month)] = y_pred
        
    X_test2017_[:, -1] = month_avgs[20]
    assert X_test2017_.shape[1] == X_test2017.shape[1]
    y_pred = lgb_model.predict(X_test2017_)
    results['201710'] = y_pred
    results['201711'] = y_pred
    results['201712'] = y_pred
    
    if j == 0:
        results_ = results.copy()
    else:
        results_ = pd.concat([results_, results], axis = 0)
    print('{}% completed'.format(round(100*(j+1)/folds)))
    
    
print('Saving predictions...')
results = results_[sample_submission.columns]
assert results.shape == sample_submission.shape
results.to_csv('submission.csv', index = False, float_format = '%.5f')
print('Done!')
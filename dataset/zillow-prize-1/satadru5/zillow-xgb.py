import numpy as np
import pandas as pd
import lightgbm as lgb
import gc

print('Loading data ...')

train = pd.read_csv('../input/train_2016_v2.csv')
prop = pd.read_csv('../input/properties_2016.csv')

for c, dtype in zip(prop.columns, prop.dtypes):	
    if dtype == np.float64:		
        prop[c] = prop[c].astype(np.float32)

df_train = train.merge(prop, how='left', on='parcelid')

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 90000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
x_train = x_train.values.astype(np.float32, copy=False)
x_valid = x_valid.values.astype(np.float32, copy=False)

#d_train = lgb.Dataset(x_train, label=y_train)
#d_valid = lgb.Dataset(x_valid, label=y_valid)

import xgboost as xgb

# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 520, 
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.93,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    #'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(x_train, y_train)


# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=1500, # increase to have better results (~700)
                  early_stopping_rounds=50,
                   verbose_eval=10, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print('num_boost_rounds=' + str(num_boost_rounds))

#num_boost_rounds = 1250
# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

print("Prepare for the prediction ...")
sample = pd.read_csv('../input/sample_submission.csv')
sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')
del sample, prop; gc.collect()
x_test = df_test[train_columns]
del df_test; gc.collect()
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
x_test = x_test.values.astype(np.float32, copy=False)

print("Start prediction ...")
# num_threads > 1 will predict very slow in kernal
#clf.reset_parameter({"num_threads":1})
dtest = xgb.DMatrix(x_test)
p_test = model.predict(dtest)

del x_test; gc.collect()

print("Start write result ...")
sub = pd.read_csv('../input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

sub.to_csv('lgb_starter_test_v2.csv', index=False, float_format='%.4f')
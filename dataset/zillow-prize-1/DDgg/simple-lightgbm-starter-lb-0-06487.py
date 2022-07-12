import numpy as np
import pandas as pd
import lightgbm as lgb
import gc

print('Loading data ...')

train = pd.read_csv('../input/train_2016.csv', low_memory=False)
prop = pd.read_csv('../input/properties_2016.csv', low_memory=False)

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

# total of 2,985,217 entries // 90000 original
split = 90000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
x_train = x_train.values.astype(np.float32, copy=False)
x_valid = x_valid.values.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)

# parameter settings, not all effective use GPU for parameter tuning
# https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.md
# https://github.com/Microsoft/LightGBM/blob/master/docs/GPU-Performance.md

params = {}
params['max_bin'] = 10           # influences memory, speed, accuracy
params['learning_rate'] = 0.002  # shrinkage_rate
params['boosting_type'] = 'gbdt' #
params['objective'] = 'regression'
params['metric'] = 'l2'          # or 'l2' or 'mae'
params['sub_feature'] = 0.4      # feature_fraction 
params['bagging_fraction'] = 0.9 # sub_row, will select X percent for bagging
params['bagging_freq'] = 20      # how often to bag
params['num_leaves'] = 60        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf use this to deal with over-fit
params['early_stopping'] = 200   # round for early stopping
params['metric_freq'] = 10       # frequence for metric output, does not work in Python

# train for 800 rounds with early stopping and parameters above
watchlist = [d_valid]
clf = lgb.train(params, d_train, 800, watchlist)

del d_train, d_valid; gc.collect()
del x_train, x_valid; gc.collect()

print("Prepare for the prediction ...")
sample = pd.read_csv('../input/sample_submission.csv', low_memory=False)
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
clf.reset_parameter({"num_threads":1})
p_test = clf.predict(x_test)

del x_test; gc.collect()

print("Start write result ...")
sub = pd.read_csv('../input/sample_submission.csv', low_memory=False)
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

sub.to_csv('lgb_starter.csv', index=False, float_format='%.4f')
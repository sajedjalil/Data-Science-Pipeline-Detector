import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
from sklearn import preprocessing

print('Loading data ...')

train = pd.read_csv('../input/train_2016_v2.csv', low_memory=False)
prop = pd.read_csv('../input/properties_2016.csv', low_memory=False)

for c, dtype in zip(prop.columns, prop.dtypes):	
    if dtype == np.float64:		
        prop[c] = prop[c].astype(np.float32)

merged_train_data = train.merge(prop, how='left', on='parcelid')

merged_data_means = merged_train_data.mean(axis=0)
merged_train_data = merged_train_data.fillna(merged_data_means)
merged_train_data = merged_train_data.fillna(-999.0)


objectTypeFeatures = ['hashottuborspa','propertycountylandusecode','propertyzoningdesc','fireplaceflag','taxdelinquencyflag']

for objectTypeFeature in objectTypeFeatures:
    label = preprocessing.LabelEncoder()
    label.fit(list(merged_train_data[objectTypeFeature].values))     
    merged_train_data[objectTypeFeature] = label.transform(list(merged_train_data[objectTypeFeature].values))

df_train = merged_train_data

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

del df_train; gc.collect()


split = 60000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
x_train = x_train.values.astype(np.float32, copy=False)
x_valid = x_valid.values.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)


params = {}
params['max_bin'] = 100           # influences memory, speed, accuracy
params['learning_rate'] = 0.002  # shrinkage_rate
params['boosting_type'] = 'gbdt' 
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mae'
params['sub_feature'] = 0.3     # feature_fraction 
params['bagging_fraction'] = 0.9 # sub_row, will select X percent for bagging
params['bagging_freq'] = 20      # how often to bag
params['num_leaves'] = 60        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf use this to deal with over-fit
params['metric_freq'] = 10       # frequence for metric output, does not work in Python

watchlist = [d_valid]
clf = lgb.train(params, d_train, 600, watchlist)

del d_train, d_valid; gc.collect()
del x_train, x_valid; gc.collect()


import xgboost as xgb
df_train_xgb = merged_train_data

x_train_xgb = df_train_xgb.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)

y_train_xgb = df_train_xgb['logerror'].values

d_train_xgb = xgb.DMatrix(x_train_xgb, y_train_xgb)

params = {}
params["eval_metric"] = "mae"
params["eta"] = 0.03
params["lambda"] = 0.8
params["max_depth"] = 5
params["subsample"] = 0.9
params["base_score"] = 0.06982

plist = list(params.items())

num_rounds = 150
xgb_model = xgb.train(plist, d_train_xgb, num_boost_round=num_rounds)


# Predictions
dtest = xgb.DMatrix(xtest)

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
pred_xgb = xgb_model.predict(x_test)
pred_final = 0.7*p_test + (1-0.7)*pred_xgb

del x_test; gc.collect()

print("Start write result ...")
sub = pd.read_csv('../input/sample_submission.csv', low_memory=False)
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = pred_final

sub.to_csv('lgb_starter.csv', index=False, float_format='%.4f')
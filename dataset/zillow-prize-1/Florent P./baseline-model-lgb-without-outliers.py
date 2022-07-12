import numpy as np
import pandas as pd
import lightgbm as lgb
import time
import gc
from sklearn.preprocessing import LabelEncoder

def remove_logerror_outliers(data):
    assert 'logerror' in data.columns, 'Data provided has no logerror column'
    print('...input shape' + str(data.shape))
    upl = np.percentile(data.logerror.values, 99)
    lol = np.percentile(data.logerror.values, 1)
    result = data[(data['logerror'] > lol) & (data['logerror'] < upl)]
    print('...output shape' + str(result.shape))
    return result

TO_BOOL = ['hashottuborspa','fireplaceflag']
TO_STRING = ['propertycountylandusecode', 'propertyzoningdesc']

def convert_to_bool(data, columns):
    if len(columns) != 0:
        for c in columns:
            data[c] = (data[c] == True)
    return data


def replace_nan(data):
    data['taxdelinquencyyear'] = data['taxdelinquencyyear'].fillna(0)
    data['taxdelinquencyflag'] = data['taxdelinquencyflag'].fillna('N')
    data['taxdelinquencyflag'] = LabelEncoder().fit_transform(data['taxdelinquencyflag'])
    return data


print('Loading data ...')
start = time.time()
train = pd.read_csv('../input/train_2016.csv')
prop = pd.read_csv('../input/properties_2016.csv')

for c, dtype in zip(prop.columns, prop.dtypes):	
    if dtype == np.float64:		
        prop[c] = prop[c].astype(np.float32)

raw = train.merge(prop, how='left', on='parcelid')
del train; gc.collect();
print('...Time elapsed ' + str(np.floor(time.time()-start)) + 's')

print('Data prep ...')
start = time.time()
df_train = replace_nan(convert_to_bool(remove_logerror_outliers(raw), TO_BOOL))
print('...Time elapsed ' + str(np.floor(time.time()-start)) + 's')
print(df_train.shape)


print('Training ...')
start = time.time()
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 85000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
x_train = x_train.values.astype(np.float32, copy=False)
x_valid = x_valid.values.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)

params = {}
params['learning_rate'] = 0.002
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mae'
params['sub_feature'] = 0.5
params['num_leaves'] = 60
params['min_data'] = 500
params['min_hessian'] = 1

watchlist = [d_valid]
clf = lgb.train(params, d_train, 500, watchlist)

del d_train, d_valid; gc.collect()
del x_train, x_valid; gc.collect()
print('...Time elapsed ' + str(np.floor(time.time()-start)) + 's')

print('Predicting on testset ...')
start = time.time()
sample = pd.read_csv('../input/sample_submission.csv')
sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')
del sample, prop; gc.collect()
x_test = df_test[train_columns]
del df_test; gc.collect()
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
x_test = x_test.values.astype(np.float32, copy=False)

clf.reset_parameter({"num_threads":1})
p_test = clf.predict(x_test)
print('...Time elapsed ' + str(np.floor(time.time()-start)) + 's')
del x_test; gc.collect()

print("Writing results...")
start = time.time()
sub = pd.read_csv('../input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

sub.to_csv('lgb_test01.csv', index=False, float_format='%.4f')
print('...Time elapsed ' + str(np.floor(time.time()-start)) + 's')
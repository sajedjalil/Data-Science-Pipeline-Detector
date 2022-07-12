import datetime
import gc
import lightgbm as lgb
import numpy as np 
import pandas as pd 
import random
import time

from sklearn.preprocessing import LabelEncoder

random.seed(0)
np.random.seed(0)

print('Loading data ...')
train = pd.read_csv('../input/train_2016.csv')
prop = pd.read_csv('../input/properties_2016.csv')

print('Mixed types: ' + str(prop.columns.values[[22,32,34,49,55]]))
mt_cols = ['hashottuborspa', 'propertycountylandusecode', \
    'propertyzoningdesc', 'fireplaceflag', 'taxdelinquencyflag']
prop[mt_cols] = prop[mt_cols].astype('str')

mu = np.mean(train.logerror.values)
si = np.std(train.logerror.values)
print('Mean logerror = ' + str(mu))
print('STD logerror = ' + str(si))

# Preprocess properties
for c, dtype in zip(prop.columns, prop.dtypes):	
    if dtype == np.float64:		
        prop[c] = prop[c].astype(np.float32)

cols_to_remove = []
for c in prop.columns:
    if len(prop[c].unique()) == 1:
        cols_to_remove.append(c)
print('Columns to remove: ' + str(cols_to_remove))
prop = prop.drop(cols_to_remove, axis=1)


prop['rawcensustractandblock'] = prop['rawcensustractandblock'].astype('str')
prop['propertyzoningdesc'] = prop['propertyzoningdesc'].astype('str')

prop['raw_census_tract'] = prop.rawcensustractandblock.apply(lambda x: \
    str(x).split('.')[0] if len(str(x).split('.')) > 0 else 'empty')
prop['block_id'] = prop.rawcensustractandblock.apply(lambda x: \
    str(x).split('.')[1] if len(str(x).split('.')) > 1 else 'empty')

prop['propertyzoningdesc_1l'] = prop.propertyzoningdesc.apply(lambda x: \
    str(x)[:1] if len(str(x).split('.')) > 0 else 'empty')
prop['propertyzoningdesc_2l'] = prop.propertyzoningdesc.apply(lambda x: \
    str(x)[:2] if len(str(x).split('.')) > 0 else 'empty')

cols = ['propertyzoningdesc', 'propertyzoningdesc_1l', \
        'propertyzoningdesc_2l', 'propertycountylandusecode', \
        'taxdelinquencyflag', 'propertylandusetypeid', 'hashottuborspa', \
        'raw_census_tract', 'block_id'] 
prop[cols] = prop[cols].astype('str')

prop[cols] = prop[cols].fillna(value='empty')
lbl_buf = []
for i, c in enumerate(cols):
    lbl_buf.append(LabelEncoder())
    lbl_buf[i].fit(list(prop[c].values))

mean_latitude_regionidzip = prop.groupby('regionidzip').latitude.mean()
mean_longitude_regionidzip = prop.groupby('regionidzip').longitude.mean()

mean_latitude_regionidcity = prop.groupby('regionidcity').latitude.mean()
mean_longitude_regionidcity = prop.groupby('regionidcity').longitude.mean()

print(mean_latitude_regionidcity)
print(mean_longitude_regionidcity)
# # prop.merge(mean_latitude_regionidzip, on='regionidzip', inplace=True)
# m_lat_d = {}
# m_lon_d = {}
# for i in mean_latitude_regionidzip.index:
#     m_lat_d[i] = mean_latitude_regionidzip[i]
#     m_lon_d[i] = mean_longitude_regionidzip[i]

df_train = train.merge(prop, how='left', on='parcelid')

# Process columns, apply LabelEncoder to categorical features
def extract_features(df, lbl_buf):
    # fmt = '%Y-%m-%d'
    # df['timestamp'] = df.transactiondate.apply(lambda x: time.mktime(datetime.datetime.strptime(x, fmt).timetuple()))
    # df['month'] = df.transactiondate.apply(lambda x: int(x.split('-')[1]))
    # df['day'] = df.transactiondate.apply(lambda x: int(x.split('-')[2]))
    df['roomcnt'] = df.bathroomcnt + df.bedroomcnt
    df['nanscnt'] = df.isnull().sum(axis=1)
    df['zeroscnt'] = (df == 0).astype(int).sum(axis=1)
    df['nanzerocnt'] = df['nanscnt'] + df['zeroscnt']
    # df['emptyscnt'] = (df == 'empty').astype(int).sum(axis=1)
    df['propertyzoningdesc_star'] = df.propertyzoningdesc.apply(lambda x: 1 if '*' in str(x) else 0)
    # df['distance_from_regionidzip_center'] = np.sqrt(np.power(df.latitude - \
    #     mean_latitude_regionidzip.get(df.regionidzip), 2) + \
    #     np.power(df.longitude - mean_longitude_regionidzip.get(df.regionidzip), 2))
    # df['d_from_regionidzip_mean'] = df.apply(lambda x: np.sqrt( \
    #     np.power(x['latitude'] - m_lat_d[x.regionidzip], 2) + \
    #     np.power(x['longitude'] - m_lon_d[x.regionidzip], 2)), axis=1)
    
    for i, c in enumerate(cols):
        df[c] = lbl_buf[i].transform(list(df[c].values))
        
    df.fillna(value=-1, inplace=True)
    df['zeroscnt'] = (df == 0).astype(int).sum(axis=1)
    return df

df_train = extract_features(df_train, lbl_buf)
# print(df_train[['nanscnt', 'propertyzoningdesc_star', 'propertyzoningdesc_2l', 'zeroscnt']])
# print(df_train[['distance_from_regionidzip_center']])

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
y_train = df_train['logerror'].values
print('x_train.shape = ' + str(x_train.shape) + ', y_train.shape = ' + str(y_train.shape))

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 90000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
x_train = x_train.values.astype(np.float32, copy=False)
x_valid = x_valid.values.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)
del x_train, x_valid; gc.collect()

params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l2'    
params['sub_feature'] = 0.5     
params['bagging_fraction'] = 0.85 
params['bagging_freq'] = 40
params['num_leaves'] = 512     
params['min_data'] = 500
params['min_hessian'] = 0.05

watchlist = [d_valid]
clf = lgb.train(params, d_train, 400, watchlist)

del d_train, d_valid; gc.collect()

print("Prepare for the prediction...")
sample = pd.read_csv('../input/sample_submission.csv')
sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')
m = 10
df_test['transactiondate'] = '2016-' + str(m) + '-15'
df_test = extract_features(df_test, lbl_buf)
del sample, prop; gc.collect()
x_test = df_test[train_columns]
del df_test; gc.collect()
print('x_test.shape = ' + str(x_test.shape))
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
x_test = x_test.values.astype(np.float32, copy=False)

print("Start prediction...")
# num_threads > 1 will predict very slow in kernal
clf.reset_parameter({"num_threads": 1})
p_test = clf.predict(x_test)
print('Mean predictions = ' + str(np.mean(p_test)))
print('STD of predictions = ' + str(np.std(p_test)))

p_test = p_test - np.mean(p_test) + mu

del x_test; gc.collect()

print("Start write result...")
sub = pd.read_csv('../input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

sub.to_csv('submission.csv', index=False, float_format='%.4f')


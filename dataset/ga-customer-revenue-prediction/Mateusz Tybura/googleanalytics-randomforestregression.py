# Importing numpy and random
import numpy as np
import random

# Setting the randomness
seed = 2018
PYTHONHASHSEED = seed
random.seed(seed)
np.random.seed(seed)

# Setting kaggle kernels number of threads
import os
os.environ['OMP_NUM_THREADS'] = '4'

# Other imports like pandas and classifier
import time
import json
import pandas as pd
from pandas.io.json import json_normalize
import gc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

# column types
start_cols = { 'channelGrouping': 'str', 'date': 'str', 'fullVisitorId': 'str', 'sessionId': 'str', 
    'socialEngagementType': 'str', 'channelGrouping': 'str', 'visitId': 'str', 'visitNumber': 'int64',
    'visitStartTime': 'int64', 'device': 'str', 'geoNetwork': 'str', 'totals': 'str', 'trafficSource': 'str'}
    
remaining_cols = {'device.browser': 'str', 'device.browserSize': 'str', 'device.browserVersion': 'str', 
    'device.deviceCategory': 'str', 'device.flashVersion': 'str', 'device.isMobile': 'bool', 
    'device.language': 'str', 'device.mobileDeviceBranding': 'str', 'device.mobileDeviceInfo': 'str', 
    'device.mobileDeviceMarketingName': 'str', 'device.mobileDeviceModel': 'str', 
    'device.mobileInputSelector': 'str', 'device.operatingSystem': 'str', 
    'device.operatingSystemVersion': 'str', 'device.screenColors': 'str', 
    'device.screenResolution': 'str', 'geoNetwork.city': 'str', 'geoNetwork.cityId': 'str', 
    'geoNetwork.continent': 'str', 'geoNetwork.country': 'str', 'geoNetwork.latitude': 'str', 
    'geoNetwork.longitude': 'str', 'geoNetwork.metro': 'str', 'geoNetwork.networkDomain': 'str', 
    'geoNetwork.networkLocation': 'str', 'geoNetwork.region': 'str', 'geoNetwork.subContinent': 'str', 
    'totals.bounces': 'int64', 'totals.hits': 'int64', 'totals.newVisits': 'int64', 'totals.pageviews': 'float32', 
    'totals.transactionRevenue': 'float32', 'totals.visits': 'int64', 'trafficSource.adContent': 'str', 
    'trafficSource.adwordsClickInfo.adNetworkType': 'str', 'trafficSource.adwordsClickInfo.criteriaParameters': 'str', 
    'trafficSource.adwordsClickInfo.gclId': 'str', 'trafficSource.adwordsClickInfo.isVideoAd': 'bool', 
    'trafficSource.adwordsClickInfo.page': 'float32', 'trafficSource.adwordsClickInfo.slot': 'str', 
    'trafficSource.campaign': 'str', 'trafficSource.campaignCode': 'str', 'trafficSource.isTrueDirect': 'bool', 
    'trafficSource.keyword': 'str', 'trafficSource.medium': 'str', 'trafficSource.referralPath': 'str', 
    'trafficSource.source': 'str'}


# data loading function
# from https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields/notebook
def load_df(csv_path='../input/train.csv', low_memory=False, nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype=start_cols, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
    
# loading data
train = load_df()
test = load_df('../input/test.csv')

# data insight
#print(list(train.columns.values))
#print(train.head())
#print(test.head())
gc.collect()

# Fixing multiple different types of NAN
nans = {nl:np.nan for nl in ['not available in demo dataset', 'unknown.unknown', '(not provided)', '(not set)']}
train.replace(nans, inplace=True)
test.replace(nans, inplace=True)

# Filling NA's with 0 or True
train['totals.transactionRevenue'].fillna(0,inplace=True)
train['trafficSource.isTrueDirect'].fillna(0,inplace=True)
train['totals.newVisits'].fillna(0,inplace=True)
train['totals.bounces'].fillna(0,inplace=True)
train['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True,inplace=True)

test['trafficSource.isTrueDirect'].fillna(0,inplace=True)
test['totals.newVisits'].fillna(0,inplace=True)
test['totals.bounces'].fillna(0,inplace=True)
test['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True,inplace=True)
gc.collect()

# Fixing column types
train_cols = list(train.columns.values)
for column in train_cols:
    print(column)
    if column in remaining_cols:
        train[column] = train[column].astype(remaining_cols[column])
        
test_cols = list(test.columns.values)
for column in test_cols:
    if column in remaining_cols:
        test[column] = test[column].astype(remaining_cols[column])
gc.collect()

# Log revenue
train['totals.transactionRevenueLog'] =  np.log1p(train['totals.transactionRevenue'])
train['totals.transactionRevenueLog'].fillna(0, inplace=True)
        
# Fixing date column
train['date_fix'] = pd.to_datetime(train['date'].astype('str'), format='%Y%m%d')
test['date_fix'] = pd.to_datetime(test['date'].astype('str'), format='%Y%m%d')

# Date features
train['year'], train['month'], train['day'], train['week']  = train['date_fix'].apply(lambda x: x.year).astype('int64'), train['date_fix'].apply(lambda x: x.month).astype('int64'), train['date_fix'].apply(lambda x: x.day).astype('int64'), train['date_fix'].apply(lambda x: x.week).astype('int64')
test['year'], test['month'], test['day'], test['week']  = test['date_fix'].apply(lambda x: x.year).astype('int64'), test['date_fix'].apply(lambda x: x.month).astype('int64'), test['date_fix'].apply(lambda x: x.day).astype('int64'), test['date_fix'].apply(lambda x: x.week).astype('int64')

train['hour'] = train['date_fix'].apply(lambda x: x.hour).astype('int64')
test['hour'] = test['date_fix'].apply(lambda x: x.hour).astype('int64')

train['dow'] = train['date_fix'].apply(lambda x: x.dayofweek).astype('int8')
test['dow'] = test['date_fix'].apply(lambda x: x.dayofweek).astype('int8')

train['on_weekend'] = train['date_fix'].apply(lambda x: x.dayofweek >= 5).astype('bool')
test['on_weekend'] = train['date_fix'].apply(lambda x: x.dayofweek >= 5).astype('bool')

train['at_night'] = train['date_fix'].apply(lambda x: x.hour <= 5 or x.hour >= 21).astype('bool')
test['at_night'] = train['date_fix'].apply(lambda x: x.hour <= 5 or x.hour >= 21).astype('bool')

train.drop(['date'], axis=1, inplace=True)
test.drop(['date'], axis=1, inplace=True)
gc.collect()

# Columns with one certain value
one_value_col = train.loc[:,(train == 'not available in demo dataset').any(axis=0)].columns
for col in one_value_col:
    if train[col].nunique() <= 1:
        train.drop(columns=col, axis=1, inplace=True)
        test.drop(columns=col, axis=1, inplace=True)
gc.collect()
        
# Other bad columns
bad_cols = ['socialEngagementType', 'geoNetwork.networkDomain', 'trafficSource.adContent', 'trafficSource.adwordsClickInfo.adNetworkType', 
    'trafficSource.adwordsClickInfo.gclId', 'trafficSource.keyword', 'trafficSource.referralPath', 'trafficSource.adwordsClickInfo.slot']
train.drop(columns=bad_cols, axis=1, inplace=True)
test.drop(columns=bad_cols, axis=1, inplace=True)
gc.collect()

# Categorize
cat_cols = ['channelGrouping', 'device.operatingSystem', 'geoNetwork.region', 'geoNetwork.metro', 'geoNetwork.city', 'trafficSource.source', 
            'day', 'month', 'week', 'totals.newVisits']
for col in cat_cols:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))
    
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')
gc.collect()

# Building useful columns and fixing remaining NAN's    
use_cols = list(train.select_dtypes(['number','category', 'bool']).columns.values)
target_cols = ['totals.transactionRevenue','totals.transactionRevenueLog']

#print train.isnull().values.any()
# any remaining NAN's in number columns
num_cols = list(train.select_dtypes(['number']).columns.values)
for col in num_cols:
    train[col].fillna(0,inplace=True)
    if col not in target_cols:
        test[col].fillna(0,inplace=True)
    
for col in target_cols: #no target column in fit and predict
    use_cols.remove(col)
    
# Putting more weight on bigger revenues
# It made everything so much worse :(
#scaler = preprocessing.MinMaxScaler()
#rev_log = train['totals.transactionRevenueLog'].values.reshape(-1,1)
#scaler.fit(rev_log)
#transformed = scaler.transform(rev_log)
#weights = transformed.flatten().tolist()

# Classification
clf = RandomForestRegressor(max_depth=10, random_state=seed)
y = train['totals.transactionRevenueLog'].values
#train.drop(columns=['totals.transactionRevenue','totals.transactionRevenueLog'], axis=1, inplace=True)
#clf.fit(train[use_cols], y, weights)
clf.fit(train[use_cols], y)
gc.collect()

# Prediction
test_id = test['fullVisitorId'].values
predictions = clf.predict(test[use_cols])
predictions[predictions<0] = 0

# Building solution
solution = pd.DataFrame({ 'fullVisitorId': test_id,'PredictedLogRevenue': predictions })
solution = solution.groupby('fullVisitorId')['PredictedLogRevenue'].sum().reset_index()
print('Saving data')
solution.to_csv('answer-' + str(time.time()) + '.csv', float_format='%.8f', index=False)
print('Saved data')
gc.collect()
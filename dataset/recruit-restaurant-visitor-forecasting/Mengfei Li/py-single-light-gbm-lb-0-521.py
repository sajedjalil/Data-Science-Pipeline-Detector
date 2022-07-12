# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:49:08 2017

@author: Mengfei Li
"""

import pandas as pd
import numpy as np
import gc
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
lb = preprocessing.LabelBinarizer()

from sklearn.model_selection import train_test_split
import lightgbm as lgb

from ml_metrics import rmsle

print("Loading Data ...")
# air_reserve
df_ar = pd.read_csv('../input/air_reserve.csv')
# air_store_info
df_as = pd.read_csv('../input/air_store_info.csv')
# air_visit_data
df_av = pd.read_csv('../input/air_visit_data.csv')
# hpg_reserve
df_hr = pd.read_csv('../input/hpg_reserve.csv')
# hpg_store_info
df_hs = pd.read_csv('../input/hpg_store_info.csv')
# date_info
df_di = pd.read_csv('../input/date_info.csv')
# sample_submission
df_ss = pd.read_csv('../input/sample_submission.csv')
# store_id_relation
df_si = pd.read_csv('../input/store_id_relation.csv')

# df_test
df_test = pd.read_csv('../input/sample_submission.csv')
df_test['air_store_id'] = df_test['id'].apply(lambda x: '_'.join(x.split('_')[:2]))
df_test['visit_date'] = df_test['id'].apply(lambda x: x.split('_')[-1])
index_test = df_test['id']
del df_test['id'], df_test['visitors']

gc.collect()
print("Loading Data Compelete.")

print("=========================================================================================")
print("Data Exploring ...")
print("=========================================================================================")
print("Unique store id in different dataset :")
print("-----------------------------------------------------------------------------------------")
num_store_ar = np.unique(df_ar['air_store_id'])
print("Number of unique stores in 'df_ar' is:" + str(len(num_store_ar)))

num_store_as = np.unique(df_as['air_store_id'])
print("Number of unique stores in 'df_as' is:" + str(len(num_store_as)))

num_store_av = np.unique(df_av['air_store_id'])
print("Number of unique stores in 'df_av' is:" + str(len(num_store_av)))

print("-----------------------------------------------------------------------------------------")
num_store_in_hr = np.unique(df_hr['hpg_store_id'])
print("Number of unique stores in 'df_hr' is:" + str(len(num_store_in_hr)))

num_store_in_hs = np.unique(df_hs['hpg_store_id'])
print("Number of unique stores in 'df_hs' is:" + str(len(num_store_in_hs)))

print("-----------------------------------------------------------------------------------------")
num_store_in_test = np.unique(df_test['air_store_id'])
print("Number of unique stores in 'df_test' is:" + str(len(num_store_in_test)))

print("-----------------------------------------------------------------------------------------")
num_store_in_si = np.unique(df_si['air_store_id'])
print("Number of unique stores in 'df_test' is:" + str(len(num_store_in_si)))
print("=========================================================================================")


# =============================================================================
# remove outliers
# =============================================================================
def remove_outliers(data):
    df_0 = data.loc[data.visitors == 0]   
    q1 = np.percentile(data.visitors, 25, axis=0)
    q3 = np.percentile(data.visitors, 75, axis=0)
#    k = 5
#    k = 2.5
    k = 2.8
#    k = 2
#    k = 1.5
    iqr = q3 - q1
    df_temp = data.loc[data.visitors > q1 - k*iqr]
    df_temp = data.loc[data.visitors < q3 + k*iqr]
    frames = [df_0, df_temp]
    result = pd.concat(frames)
    return result

df_av = remove_outliers(df_av)


# =============================================================================
# df to dict for mapping and dropping
# =============================================================================
print('mapping and dropping useless information in df_hr ...')
s_1 = df_si['air_store_id']
s_2 = df_si['hpg_store_id']
a_h_map = dict(zip(s_2.values, s_1.values))
del s_1, s_2

df_hr['air_store_id'] = df_hr['hpg_store_id'].map(a_h_map)
df_hr = df_hr.drop('hpg_store_id', axis=1).dropna()


print('mapping and dropping useless information in df_hr Done!')
print("-----------------------------------------------------------------------------------------")

print('mapping and dropping useless information in df_hr ...')

df_hs['air_store_id'] = df_hs['hpg_store_id'].map(a_h_map)
df_hs = df_hs.drop('hpg_store_id', axis=1).dropna()
print('mapping and dropping useless information in df_hs Done!')
gc.collect()
print("=========================================================================================")


# =============================================================================
# handle datetime (no clock info)
# =============================================================================
print('seperating date time features ...')

time_format = '%Y-%m-%d'
def seperate_date(data):     
    # split date feature in real visit datetime
    data_time = pd.to_datetime(data.visit_date, format=time_format)
    data['Year_visit']= data_time.dt.year
    data['Month_visit'] = data_time.dt.month
    data['DayOfYear_visit'] = data_time.dt.dayofyear
    # data['DayOfMonth_visit'] = data_time.dt.day
#    data['WeekOfYear_visit'] = data_time.dt.week
    data['DayOfWeek_visit'] = data_time.dt.dayofweek
#    del data['visit_date']
    return data

seperate_date(df_av)
seperate_date(df_test)

# ------------------------------------------------------------------------------
time_format = "%Y-%m-%d %H:%M:%S"
def seperate_date(data):
    # split date feature in reservation datetime
    data_time = pd.to_datetime(data.reserve_datetime, format=time_format)
    data['Year_re']= data_time.dt.year
    data['Month_re'] = data_time.dt.month
    data['DayOfYear_re'] = data_time.dt.dayofyear
    # data['DayOfMonth_re'] = data_time.dt.day
#    data['WeekOfYear_re'] = data_time.dt.week
    data['DayOfWeek_re'] = data_time.dt.dayofweek
    data['Hour_re'] = data_time.dt.hour
#    del data['reserve_datetime']
    return data

seperate_date(df_ar)


def seperate_date(data):
    # split date feature in reservation datetime
    data_time = pd.to_datetime(data.reserve_datetime, format=time_format)
    data['Year_re_h']= data_time.dt.year
    data['Month_re_h'] = data_time.dt.month
    data['DayOfYear_re_h'] = data_time.dt.dayofyear
    # data['DayOfMonth_re_h'] = data_time.dt.day
#    data['WeekOfYear_re_h'] = data_time.dt.week
    data['DayOfWeek_re_h'] = data_time.dt.dayofweek
    data['Hour_re_h'] = data_time.dt.hour
#    del data['reserve_datetime']
    return data

seperate_date(df_hr)


time_format = "%Y-%m-%d %H:%M:%S"
def seperate_date(data):
    # split date feature in reserved visiting datetime
    data_time = pd.to_datetime(data.visit_datetime, format=time_format)
    data['Year_re_visit']= data_time.dt.year
    data['Month_re_visit'] = data_time.dt.month
    data['DayOfYear_re_visit'] = data_time.dt.dayofyear
    # data['DayOfMonth_re_visit'] = data_time.dt.day
#    data['WeekOfYear_re_visit'] = data_time.dt.week
    data['DayOfWeek_re_visit'] = data_time.dt.dayofweek
    data['Hour_re_visit'] = data_time.dt.hour
#    del data['visit_datetime']
    return data

seperate_date(df_ar)


def seperate_date(data):
    # split date feature in reserved visiting datetime
    data_time = pd.to_datetime(data.visit_datetime, format=time_format)
    data['Year_re_visit_h']= data_time.dt.year
    data['Month_re_visit_h'] = data_time.dt.month
    data['DayOfYear_re_visit_h'] = data_time.dt.dayofyear
    # data['DayOfMonth_re_visit_h'] = data_time.dt.day
    data['WeekOfYear_re_visit_h'] = data_time.dt.week
    data['DayOfWeek_re_visit_h'] = data_time.dt.dayofweek
    data['Hour_re_visit_h'] = data_time.dt.hour
#    del data['visit_datetime']
    return data

seperate_date(df_hr)

print('seperating date time features done! ...')
gc.collect()
print("=========================================================================================")


# =============================================================================
# label encoding
# =============================================================================
print('label encoding ...')

le.fit(df_as['air_genre_name'])
df_as['air_genre_name'] = le.fit_transform(df_as['air_genre_name'])

le.fit(df_as['air_area_name'])
df_as['air_area_name'] = le.fit_transform(df_as['air_area_name'])

le.fit(df_hs['hpg_genre_name'])
df_hs['hpg_genre_name'] = le.fit_transform(df_hs['hpg_genre_name'])

le.fit(df_hs['hpg_area_name'])
df_hs['hpg_area_name'] = le.fit_transform(df_hs['hpg_area_name'])



le.fit(df_as['air_store_id'])


df_ar['air_store_id'] = le.transform(df_ar['air_store_id'])
df_as['air_store_id'] = le.transform(df_as['air_store_id'])
df_av['air_store_id'] = le.transform(df_av['air_store_id'])
df_hr['air_store_id'] = le.transform(df_hr['air_store_id'])
df_hs['air_store_id'] = le.transform(df_hs['air_store_id'])

df_test['air_store_id'] = le.transform(df_test['air_store_id'])


print('label encoding done !')
gc.collect()
print("=========================================================================================")



# =============================================================================
# Merge dataset
# =============================================================================
features_to_drop = [
        'air_store_id__'
        ]

def merge_df(data, data_to_join):
    # merge dataframes        
    data = data.join(data_to_join, on='air_store_id', rsuffix='__', how='left')   
    return data

def fix_data(data):
    # drop __ data    
    for feature in features_to_drop:
        del data[feature]
    return data

# Merge to df_train
print('merging dataframes ...')
df_train = merge_df(df_av, df_ar)
df_train = merge_df(df_train, df_as)

df_hr['reserve_visitors_hr'] = df_hr['reserve_visitors'] 
del df_hr['reserve_visitors'] 

df_hs['latitude_hr'] = df_hs['latitude'] 
del df_hs['latitude'] 

df_hs['longitude_hr'] = df_hs['longitude'] 
del df_hs['longitude'] 

df_train = merge_df(df_train, df_hs)
df_train = merge_df(df_train, df_hr)
gc.collect()
fix_data(df_train)

# Merge to df_test

df_test = merge_df(df_test, df_ar)
df_test = merge_df(df_test, df_as)

df_test = merge_df(df_test, df_hs)
df_test = merge_df(df_test, df_hr)
gc.collect()
fix_data(df_test)


print('merging dataframes done!')
gc.collect()
print("=========================================================================================")


# =============================================================================
# add holiday feature (for the visiting day)
# =============================================================================
df_di['visit_date'] = df_di['calendar_date']
del df_di['calendar_date'] 

def add_is_holiday(data):
    # merge dataframes        
    data = data.merge(df_di, on='visit_date', how='left')
    del data['day_of_week']
    return data

df_train = add_is_holiday(df_train)
df_test = add_is_holiday(df_test)

# =============================================================================
# drop date-time-hour info
# =============================================================================
def drop_datetime_info(data):
    del data['visit_date'], data['visit_datetime'], data['reserve_datetime'], data['visit_datetime__'], data['reserve_datetime__']
#    del data['visit_date'], data['visit_datetime'], data['reserve_datetime']
    return data
df_train = drop_datetime_info(df_train)

def drop_datetime_info(data):
    del data['visit_date'], data['visit_datetime'], data['reserve_datetime'], data['visit_datetime__'], data['reserve_datetime__']
#    del data['visit_date'], data['visit_datetime'], data['reserve_datetime']
    return data
df_test = drop_datetime_info(df_test)



# =============================================================================
# autoclean
# =============================================================================
#df_train_clean = autoclean(df_train)
#df_test_clean = autoclean(df_test)
#
train = df_train.fillna(-1)
test = df_test.fillna(-1)
#
# =============================================================================
# shuffle dataset
# =============================================================================
from sklearn.utils import shuffle
train =  shuffle(train, random_state=21)


X_train, X_valid = train_test_split(train, test_size=0.05, random_state=42, shuffle=False)

X = X_train.drop(['visitors'], axis=1)
y = np.log1p(X_train['visitors'].values)
d_train = lgb.Dataset(X, y)

X = X_valid.drop(['visitors'], axis=1)
y = np.log1p(X_valid['visitors'].values)
d_valid = lgb.Dataset(X, y)

watchlist = [d_train, d_valid]

print('Training LGBM model...')
params = {}
params['application'] = 'regression'
params['boosting'] = 'gbdt'
params['learning_rate'] = 0.01
params['num_leaves'] = 32
params['min_sum_hessian_in_leaf'] = 1e-2
params['min_gain_to_split'] = 0

params['bagging_fraction'] = 0.8
params['feature_fraction'] = 0.8
params['num_threads'] = 4
params['metric'] = 'rmse'

lgb_model1 = lgb.train(params, train_set=d_train, num_boost_round=50000, valid_sets=watchlist, \
verbose_eval=10)

test_probs = lgb_model1.predict(test)
test_probs = np.expm1(test_probs)

result = pd.DataFrame({"id": index_test, "visitors": test_probs})
    
result.to_csv('LGB_sub.csv', index=False)
    
    # gbm.save_model(r"..\output\models\LGB_"+str(file_name)+'.model')
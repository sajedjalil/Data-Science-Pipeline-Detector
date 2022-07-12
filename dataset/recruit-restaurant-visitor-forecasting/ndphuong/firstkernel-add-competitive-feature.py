# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import ensemble, neighbors, preprocessing, metrics
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5

path = r"../input"    
data = {
    'tra': pd.read_csv(path + '/air_visit_data.csv'),
    'as': pd.read_csv(path + '/air_store_info.csv'),
    'hs': pd.read_csv(path + '/hpg_store_info.csv'),
    'ar': pd.read_csv(path + '/air_reserve.csv'),
    'hr': pd.read_csv(path + '/hpg_reserve.csv'),
    'id': pd.read_csv(path + '/store_id_relation.csv'),
    'tes': pd.read_csv(path + '/sample_submission.csv'),
    'hol': pd.read_csv(path + '/date_info.csv').rename(columns={'calendar_date': 'visit_date'})
    }

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])
print("Finish load raw data!\n")
# convert datetime
# count number/average of visitors per (resto, visit date)
print("Start pre-processing data:\n")
for df in ['ar', 'hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])  # standardize the format to pandas format
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date  # keep only date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    # add information about diff (in days) between reservation date and visit date
    # apply the spread on two columns for each line ==> use apply
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    # group data by pair (resto, visit date)
    tmp = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']]
    tmp1 = tmp.sum().rename(columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors': 'rv1'})
    tmp2 = tmp.mean().rename(columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors': 'rv2'})
    data[df] = pd.merge(tmp1, tmp2,  how='inner', on=['air_store_id', 'visit_date'])

# get detail of date, month, year on TRAin data (AIR_VISIT_DATA)
data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

# get detail of date, month, year on test data
# first need to split the datetime from restoID and then do the same for train data
# apply the mapping for each elem of a columns ==> use map function
data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])   # keep the last elem after "_"
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))   # keep all elems but 3rd(2)
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

# get unique store list
unique_stores = data['tes']['air_store_id'].unique()
"""
prepare the (output) format to do the model training
"""
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

"""
The seasonality: number of visitors per date depends on which date it is: Monday, Tuesday,etc..., Saturday, Sunday
==> need to aggregate the distribution across "dow" (day of week)
"""
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].min().rename(columns={'visitors': 'min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].mean().rename(columns={'visitors': 'mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].median().rename(columns={'visitors': 'median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].max().rename(columns={'visitors': 'max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].count().rename(columns={'visitors': 'count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])

# competitive feature
tmp = data['tra'].groupby(['air_store_id'], as_index=False)['visitors'].sum().rename(columns={'visitors': 'total_visitors_per_resto'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id'])
tmp = data['tra'].groupby(['air_store_id', 'month'], as_index=False)['visitors'].sum().rename(columns={'visitors': 'total_visitors_per_resto_per_month'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id'])
tmp = data['tra'].groupby(['month'], as_index=False)['visitors'].sum().rename(columns={'visitors': 'total_visitors_all_resto_per_month'})
stores = pd.merge(stores, tmp, how='left', on=['month'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].sum().rename(columns={'visitors': 'total_visitors_per_resto_per_dow'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['dow'], as_index=False)['visitors'].sum().rename(columns={'visitors': 'total_visitors_all_resto_per_dow'})
stores = pd.merge(stores, tmp, how='left', on=['dow'])
total_visitors_all_resto = data['tra']['visitors'].sum()
stores['compete_1'] = stores.apply(lambda r: r['total_visitors_per_resto']/total_visitors_all_resto, axis=1)
stores['compete_2'] = stores.apply(lambda r: r['total_visitors_per_resto_per_month']/r['total_visitors_all_resto_per_month'], axis=1)
stores['compete_3'] = stores.apply(lambda r: r['total_visitors_per_resto_per_dow']/r['total_visitors_all_resto_per_dow'], axis=1)

# complete the information from AS table:
stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])
# need to re-encode the string format
stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/', ' ')))
stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-', ' ')))
lbl = preprocessing.LabelEncoder()  # to encode string into numeric (keep the number of unique values)
for i in range(10):
    stores['air_genre_name'+str(i)] = lbl.fit_transform(stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
    stores['air_area_name'+str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

print("End pre-processing data!\n")
# create training and test set
print("Start creating training set and test set:\n")
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date'])
# merge with stores information
train = pd.merge(train, stores, how='left', on=['air_store_id', 'dow'])
test = pd.merge(test, stores, how='left', on=['air_store_id', 'dow'])

"""
for df in ['ar', 'hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id', 'visit_date'])
    test = pd.merge(test, data[df], how='left', on=['air_store_id', 'visit_date'])
"""
train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

# using comprehensive list to define the features used for training model
col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date', 'visitors',
'latitude','longitude','day_of_week','median_visitors']]
# fill na value
train = train.fillna(-1)    # using backfill
test = test.fillna(-1)

print("End creating training set and test set!\n")
# define models
print("Start defining models:\n")
model1 = ensemble.GradientBoostingRegressor(learning_rate=0.2, random_state=3, n_estimators=100,
                                            subsample=0.8, max_depth=10)
model2 = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=4)
#model3 = ensemble.RandomForestRegressor(n_estimators = 200, criterion="mse", max_features = 0.7,
                                        #min_samples_leaf=200, random_state=5, warm_start = True)

print("End defining models!\n")
# train models
print("Start training models:\n")
model1.fit(train[col], np.log1p(train["visitors"].values))
model2.fit(train[col], np.log1p(train["visitors"].values))
#model3.fit(train[col], np.log1p(train["visitors"].values))
print("End training models!\n")
# predict on train set
print("Start predicting on training set:\n")
pred1 = model1.predict(train[col])
pred2 = model2.predict(train[col])
#pred3 = model3.predict(train[col])

print("RMSE GradientBoostRegressor: ", RMSLE(np.log1p(train["visitors"].values), pred1))
print("RMSE KNeighbor: ", RMSLE(np.log1p(train["visitors"].values), pred2))
#print("RMSE RandomForest: ", RMSLE(np.log1p(train["visitors"].values), pred3))

# predict on test set
test_pred1 = model1.predict(test[col])
test_pred2 = model2.predict(test[col])
#test_pred3 = model3.predict(test[col])

# aggregate result from two models
test['visitors'] = 0.8*test_pred1 + 0.2*test_pred2 #+ 0.4*test_pred3
test['visitors'] = np.expm1(test['visitors']).clip(lower=0.)
sub1 = test[['id', 'visitors']].copy()
sub1[['id', 'visitors']].to_csv('submission.csv', index=False)
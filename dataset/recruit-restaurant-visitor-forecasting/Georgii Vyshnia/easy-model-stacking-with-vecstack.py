import numpy as np
import pandas as pd

from sklearn import *
import datetime as dt
# vecstack is not available in Kaggle kernels - it will raise an error here
# you should install vecstack into your Python first, to be able to run this code locally
from vecstack import stacking 


def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred) ** 0.5

start_time = dt.datetime.now()
print("Started at ", start_time)

data = {
    'tra': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'tes': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date': 'visit_date'})
}

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

for df in ['ar', 'hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days,
                                                       axis=1)
    data[df] = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[
        ['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime': 'visit_date'})
    print(data[df].head())

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i] * len(unique_stores)}) for i in range(7)],
                   axis=0, ignore_index=True).reset_index(drop=True)

# sure it can be compressed...
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].min().rename(
    columns={'visitors': 'min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].mean().rename(
    columns={'visitors': 'mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].median().rename(
    columns={'visitors': 'median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].max().rename(
    columns={'visitors': 'max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].count().rename(
    columns={'visitors': 'count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])
lbl = preprocessing.LabelEncoder()
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date'])

train = pd.merge(data['tra'], stores, how='left', on=['air_store_id', 'dow'])
test = pd.merge(data['tes'], stores, how='left', on=['air_store_id', 'dow'])

for df in ['ar', 'hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id', 'visit_date'])
    test = pd.merge(test, data[df], how='left', on=['air_store_id', 'visit_date'])

print(train.describe())
print(train.head())

train = train.fillna(-1)
test = test.fillna(-1)

col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date', 'visitors']]

X_train = train[col]
y = np.log1p(train['visitors'].values)
X_test = test[col]

# casting X-dataframes to numpy arrays
X_train = X_train.values
X_test = X_test.values

print("Finished pre-processed data load at ", dt.datetime.now())

# Configure models
seed = 1914

lr = linear_model.LinearRegression(n_jobs=-1)
etc = ensemble.ExtraTreesRegressor(n_estimators=225, max_depth=5, n_jobs=-1, random_state=seed)
knn = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=4)

rf = ensemble.RandomForestRegressor(random_state=seed, n_jobs=-1,
                          n_estimators=100, max_depth=3)

print("Finished setting up regressors at ", dt.datetime.now())

# Initialize 1-st level models.
models = [rf, etc, knn]

# Compute stacking features
S_train, S_test = stacking(models, X_train, y, X_test,
    regression = True, metric = RMSLE, n_folds = 4,
    shuffle = True, random_state = seed, verbose = 2)

# Initialize 2-nd level model
model = lr

# Fit 2-nd level model
model = model.fit(S_train, y)

# Predict
y_test_pred = model.predict(S_test)


# Create submission file
sub = pd.DataFrame()
sub['id'] = test['id']
sub['visitors'] = np.expm1(y_test_pred) # .clip(lower=0.)
sub.to_csv('stacking.csv', float_format='%.4f', index=False)

print('We are done. That is all, folks!')
finish_time = dt.datetime.now()
print("Started at ", finish_time)
elapsed = finish_time - start_time
print("Elapsed time: ", elapsed)
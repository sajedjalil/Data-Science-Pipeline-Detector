# GBM prediction

# inspirations:
# - feature engineering and pre-processing reused from https://www.kaggle.com/the1owl/surprise-me/

import numpy as np
import pandas as pd
from sklearn import *
import datetime as dt

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

col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date', 'visitors']]
train = train.fillna(-1)
test = test.fillna(-1)

X = train[col]
y = pd.DataFrame()
y['visitors'] = np.log1p(train['visitors'].values)

y_test_pred = 0

print("Finished data pre-processing at ", dt.datetime.now())

# Set up folds
K = 4
kf = model_selection.KFold(n_splits = K, random_state = 1, shuffle = True)
np.random.seed(1)


# model
# set up GBM regression model
params = {'n_estimators': 100, # change to 4000 to achieve LB 0.511 - it runs too long in the kaggle kernel mode
        'max_depth': 5,
        'min_samples_split': 200, 
        'min_samples_leaf': 50,
        'learning_rate': 0.005,
        'max_features':  9,
        'subsample': 0.8,
        'loss': 'ls'}
model = ensemble.GradientBoostingRegressor(**params)
print("Finished setting up CV folds and regressor at ", dt.datetime.now())
# Run CV

print("Started CV at ", dt.datetime.now())
for i, (train_index, test_index) in enumerate(kf.split(train)):
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index, :].copy(), X.iloc[test_index, :].copy()
    X_test = test[col]
    print("\nFold ", i)

    fit_model = model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    print('RMSE GBM Regressor, fold ', i, ': ', RMSLE(y_valid, pred))
    print('Prediction length on validation set, GBM Regressor, fold ', i, ': ', len(pred))
    # Accumulate test set predictions

    pred = model.predict(X_test)
    print('Prediction length on test set, GBM Regressor, fold ', i, ': ', len(pred))
    y_test_pred += pred

    del X_test, X_train, X_valid, y_train

print("Finished CV at ", dt.datetime.now())

y_test_pred /= K  # Average test set predictions
print("Finished average test set predictions at ", dt.datetime.now())

# Create submission file
sub = pd.DataFrame()
sub['id'] = test['id']
sub['visitors'] = np.expm1(y_test_pred) 
sub.to_csv('gbm_submit.csv', float_format='%.6f', index=False)

print('We are done. That is all, folks!')
finish_time = dt.datetime.now()
print("Started at ", finish_time)
elapsed = finish_time - start_time
print("Elapsed time: ", elapsed)
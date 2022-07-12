# A little memory heavy but should generate some options for anyone stuck
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np; np.random.seed(17)
from datetime import timedelta
import random; random.seed(17)
import pandas as pd
import numpy as np
from sklearn import *
from multiprocessing import *
import datetime as dt
import gc; gc.enable()
from catboost import CatBoostRegressor
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb

directory="../input/"

def add_date_features(df):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = (df["transactiondate"].dt.year - 2016)*12 + df["transactiondate"].dt.month
    df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = (df["transactiondate"].dt.year - 2016)*4 +df["transactiondate"].dt.quarter
    df.drop(["transactiondate"], inplace=True, axis=1)
    return df


cal = USFederalHolidayCalendar()
holidays = [d.date() for d in cal.holidays(start='2016-01-01', end='2017-12-31').to_pydatetime()]
business = [d.date() for d in pd.date_range('2016-01-01', '2017-12-31') if d not in pd.bdate_range('2016-01-01', '2017-12-31')]
holidays_prev = [d + timedelta(days=-1) for d in holidays]
holidays_after = [d + timedelta(days=1) for d in holidays]

#Train 16
train = pd.read_csv(directory+'train_2016_v2.csv')
train = pd.merge(train, pd.read_csv(directory+'properties_2016.csv'), how='left', on='parcelid')
train_17 = pd.read_csv(directory+'train_2017.csv')
train_17 = pd.merge(train_17, pd.read_csv(directory+'properties_2017.csv'), how='left', on='parcelid')
train = pd.concat((train, train_17), axis=0, ignore_index=True).reset_index(drop=True)
del train_17; gc.collect();

ecol = [c for c in train.columns if train[c].dtype == 'object'] + ['taxdelinquencyflag','propertycountylandusecode','propertyzoningdesc','parcelid','ParcelId','logerror','transactiondate']
col = [c for c in train.columns if c not in ['taxdelinquencyflag','propertycountylandusecode','propertyzoningdesc','parcelid','ParcelId','logerror','transactiondate']]
dcol = col.copy()
d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
one_hot = {c: list(train[c].unique()) for c in col}

#df_dd_sheets = [pd.read_excel('../input/zillow_data_dictionary.xlsx', sheetname=i) for i in range(8)]
#print(df_dd_sheets[0].head())

def transform_df(df):
    try:
        df = pd.DataFrame(df)
        df['null_vals'] = df.isnull().sum(axis=1)
        df['transactiondate'] = pd.to_datetime(df['transactiondate'])
        df['transactiondate_year'] = df['transactiondate'].dt.year
        df['transactiondate_month'] = df['transactiondate'].dt.month
        df['transactiondate_day'] = df['transactiondate'].dt.day
        df['transactiondate_dow'] = df['transactiondate'].dt.dayofweek
        df['transactiondate_wd'] = df['transactiondate'].dt.weekday
        df['transactiondate_h'] = df['transactiondate'].dt.date.map(lambda x: 1 if x in holidays else 0)
        df['transactiondate_hp'] = df['transactiondate'].dt.date.map(lambda x: 1 if x in holidays_prev else 0)
        df['transactiondate_ha'] = df['transactiondate'].dt.date.map(lambda x: 1 if x in holidays_after else 0)
        df['transactiondate_b'] = df['transactiondate'].dt.date.map(lambda x: 1 if x in business else 0)
        df['transactiondate_quarter'] = df['transactiondate'].dt.quarter
        df = df.fillna(-1.0)
        for c in dcol:
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)
            #df[c+str('_sq')] = np.power(df[c].values,2).astype(np.float32)
            #df[c+str('_sqr')] = np.square(df[c].values).astype(np.float32)
            #df[c+str('_log')] = np.log(np.abs(df[c].values) + 1)
            #df[c+str('_exp')] = np.exp(df[c].values) - 1
        for c in one_hot:
            if len(one_hot[c])>2 and len(one_hot[c]) < 10:
                for val in one_hot[c]:
                    df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    except Exception as e: 
        print(e)
    return df

def multi_transform(df):
    print('Init Shape: ', df.shape)
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close(); p.join()
    print('After Shape: ', df.shape)
    return df

def MAE(y, pred):
    #logerror=log(Zestimate)−log(SalePrice)
    return np.sum([abs(y[i]-pred[i]) for i in range(len(y))]) / len(y)

train = multi_transform(train)
col = [c for c in train.columns if c not in ecol]
train_df=train[col]


missing_perc_thresh = 0.98
exclude_missing = []
num_rows = train_df.shape[0]
for c in train_df.columns:
    num_missing = train_df[c].isnull().sum()
    if num_missing == 0:
        continue
    missing_frac = num_missing / float(num_rows)
    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)
print("We exclude: %s" % len(exclude_missing))

del num_rows, missing_perc_thresh
gc.collect();



print("Remove features with one unique value !!")
exclude_unique = []
for c in train_df.columns:
    num_uniques = len(train_df[c].unique())
    if train_df[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)
print("We exclude: %s" % len(exclude_unique))

print("Define training features !!")
exclude_other = ['parcelid', 'logerror', 'propertyzoningdesc']
train_features = []
for c in train_df.columns:
    if c not in exclude_missing \
            and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)
print("We use these for training: %s" % len(train_features))

print("Define categorial features !!")
cat_feature_inds = []
cat_unique_thresh = 1000
for i, c in enumerate(train_features):
    num_uniques = len(train_df[c].unique())
    if num_uniques < cat_unique_thresh \
            and not 'sqft' in c \
            and not 'cnt' in c \
            and not 'nbr' in c \
            and not 'number' in c:
        cat_feature_inds.append(i)

print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])

print("Replacing NaN values by -999 !!")
train_df.fillna(-999, inplace=True)


print('Catboosting...')
#temp fix to Sklearn MLK error by @Yakolle Zhang
'''
reg = linear_model.Lasso()
reg.fit(train_df, train['logerror'])

reg = linear_model.LinearRegression(n_jobs=-1)
reg.fit(train_df, train['logerror'])
'''
X_train=train_df[train_features]
y_train=train.logerror

def train_and_test(X_train,y_train,i):
    cat_model = CatBoostRegressor(
        iterations=630, learning_rate=0.03,
        depth=6, l2_leaf_reg=3,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed=i)
    cat_model.fit(
        X_train, y_train,
        cat_features=cat_feature_inds)

    print('round '+str(i)+' trained!')
    gc.collect();

    # Pred 16
    test = pd.read_csv(directory + 'sample_submission.csv')
    test_col = [c for c in test.columns]
    test = pd.merge(test, pd.read_csv(directory + 'properties_2016.csv'), how='left', left_on='ParcelId',
                    right_on='parcelid')
    test_dates = ['2016-10-01', '2016-11-01', '2016-12-01']
    test_columns = ['201610', '201611', '201612']
    for i in range(len(test_dates)):
        transactiondate = dt.date(*(int(s) for s in test_dates[i].split('-')))
        dr = pd.date_range(transactiondate, transactiondate + timedelta(days=27))
        test['transactiondate'] = np.random.choice(dr, len(test['ParcelId']))
        test = multi_transform(test)  # keep order
        # test[test_columns[i]] = reg.predict(test[col])
        test[test_columns[i]] = cat_model.predict(test[train_features])
        print('predict...', test_dates[i])

    # Pred 17
    test = test[test_col]
    test = pd.merge(test, pd.read_csv(directory + 'properties_2017.csv'), how='left', left_on='ParcelId',
                    right_on='parcelid')
    test_dates = ['2017-10-01', '2017-11-01', '2017-12-01']
    test_columns = ['201710', '201711', '201712']
    for i in range(len(test_dates)):
        transactiondate = dt.date(*(int(s) for s in test_dates[i].split('-')))
        dr = pd.date_range(transactiondate, transactiondate + timedelta(days=27))
        test['transactiondate'] = np.random.choice(dr, len(test['ParcelId']))
        test = multi_transform(test)  # keep order
        test[test_columns[i]] = cat_model.predict(test[train_features])
        print('predict...', test_dates[i])
    return test[test_col]

num_ensemble=5
test = pd.read_csv(directory + 'sample_submission.csv')
test_col=[c in test.columns and c!='ParcelId']
for i in tqdm(range(num_ensemble)):
    pred=train_and_test(X_train,y_train,i)[test_col]
    for c in test_col:
        test[c]+=pred[c]
test[test_col] /=num_ensemble

test.to_csv(directory+'cat_submission.csv.gz', index=False, compression='gzip', float_format='%.4f')

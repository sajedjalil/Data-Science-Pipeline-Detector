import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostRegressor
from tqdm import tqdm
import gc
import datetime as dt

print('Loading Properties ...')
properties2016 = pd.read_csv('../input/properties_2016.csv', low_memory = False)
properties2017 = pd.read_csv('../input/properties_2017.csv', low_memory = False)

print('Loading Train ...')
train2016 = pd.read_csv('../input/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
train2017 = pd.read_csv('../input/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)

def add_date_features(df):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = (df["transactiondate"].dt.year - 2016)*12 + df["transactiondate"].dt.month
    df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = (df["transactiondate"].dt.year - 2016)*4 +df["transactiondate"].dt.quarter
    df.drop(["transactiondate"], inplace=True, axis=1)
    return df

train2016 = add_date_features(train2016)
train2017 = add_date_features(train2017)

print('Loading Sample ...')
sample_submission = pd.read_csv('../input/sample_submission.csv', low_memory = False)

print('Merge Train with Properties ...')
train2016 = pd.merge(train2016, properties2016, how = 'left', on = 'parcelid')
train2017 = pd.merge(train2017, properties2017, how = 'left', on = 'parcelid')

print('Tax Features 2017  ...')
train2017.iloc[:, train2017.columns.str.startswith('tax')] = np.nan

print('Concat Train 2016 & 2017 ...')
train_df = pd.concat([train2016, train2017], axis = 0)
test_df = pd.merge(sample_submission[['ParcelId']], properties2016.rename(columns = {'parcelid': 'ParcelId'}), how = 'left', on = 'ParcelId')

del properties2016, properties2017, train2016, train2017
gc.collect();

print('Remove missing data fields ...')

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

print ("Remove features with one unique value !!")
exclude_unique = []
for c in train_df.columns:
    num_uniques = len(train_df[c].unique())
    if train_df[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)
print("We exclude: %s" % len(exclude_unique))

print ("Define training features !!")
exclude_other = ['parcelid', 'logerror','propertyzoningdesc']
train_features = []
for c in train_df.columns:
    if c not in exclude_missing \
       and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)
print("We use these for training: %s" % len(train_features))

print ("Define categorial features !!")
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

print ("Replacing NaN values by -999 !!")
train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)

print ("Training time !!")
X_train = train_df[train_features]
y_train = train_df.logerror
print(X_train.shape, y_train.shape)

test_df['transactiondate'] = pd.Timestamp('2016-12-01') 
test_df = add_date_features(test_df)
X_test = test_df[train_features]
print(X_test.shape)

num_ensembles = 5
y_pred = 0.0
for i in tqdm(range(num_ensembles)):
    model = CatBoostRegressor(
        iterations=630, learning_rate=0.03,
        depth=6, l2_leaf_reg=3,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed=i)
    model.fit(
        X_train, y_train,
        cat_features=cat_feature_inds)
    y_pred += model.predict(X_test)
y_pred /= num_ensembles

submission = pd.DataFrame({
    'ParcelId': test_df['ParcelId'],
})
test_dates = {
    '201610': pd.Timestamp('2016-09-30'),
    '201611': pd.Timestamp('2016-10-31'),
    '201612': pd.Timestamp('2016-11-30'),
    '201710': pd.Timestamp('2017-09-30'),
    '201711': pd.Timestamp('2017-10-31'),
    '201712': pd.Timestamp('2017-11-30')
}
for label, test_date in test_dates.items():
    print("Predicting for: %s ... " % (label))
    submission[label] = y_pred
    
submission.to_csv('Only_CatBoost.csv', float_format='%.6f',index=False)
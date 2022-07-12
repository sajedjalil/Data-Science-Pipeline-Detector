import numpy as np
import pandas as pd

import warnings
warnings.simplefilter('ignore')

from cross_validation_framework import *

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load data

train = pd.read_csv('../input/cat-in-the-dat/train.csv', index_col='id')
test = pd.read_csv('../input/cat-in-the-dat/test.csv', index_col='id')


# Feature engineering

# Save columns that we will use for feature aggregates

fa_features = [
    'bin_0', 'bin_1',
    'nom_5', 'nom_6'
]

train_fa = train[fa_features].copy()
test_fa = test[fa_features].copy()


# OHE

ohe_features = [
    'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
    'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4',
    'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4',
    'day', 'month'
]

le_features = list(set(test.columns) - set(ohe_features))

train_part = len(train)
df = pd.get_dummies(pd.concat([train, test], axis=0), columns=ohe_features)
train = df[:train_part]
test = df[train_part:].drop('target', axis=1)
del df


# LE

# From https://www.kaggle.com/pavelvpster/ieee-fraud-eda-lightgbm-baseline

from sklearn.preprocessing import LabelEncoder


def encode_categorial_features_fit(df, columns_to_encode):
    encoders = {}
    for c in columns_to_encode:
        if c in df.columns:
            encoder = LabelEncoder()
            encoder.fit(df[c].astype(str).values)
            encoders[c] = encoder
    return encoders

def encode_categorial_features_transform(df, encoders):
    out = pd.DataFrame(index=df.index)
    for c in encoders.keys():
        if c in df.columns:
            out[c] = encoders[c].transform(df[c].astype(str).values)
    return out


categorial_features_encoders = encode_categorial_features_fit(
    pd.concat([train, test], join='outer', sort=False), le_features)

temp = encode_categorial_features_transform(train, categorial_features_encoders)
columns_to_drop = list(set(le_features) & set(train.columns))
train = train.drop(columns_to_drop, axis=1).merge(temp, how='left', left_index=True, right_index=True)
del temp

temp = encode_categorial_features_transform(test, categorial_features_encoders)
columns_to_drop = list(set(le_features) & set(test.columns))
test = test.drop(columns_to_drop, axis=1).merge(temp, how='left', left_index=True, right_index=True)
del temp


# TE

from category_encoders import TargetEncoder


te_features = [
    'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'
]

te = TargetEncoder(cols=te_features, drop_invariant=True, return_df=True, min_samples_leaf=2, smoothing=1.0)
te.fit(train[te_features], train['target'])

temp = te.transform(train[te_features])
columns_to_drop = list(set(te_features) & set(train.columns))
train = train.drop(columns_to_drop, axis=1).merge(temp, how='left', left_index=True, right_index=True)
del temp

temp = te.transform(test[te_features])
columns_to_drop = list(set(te_features) & set(test.columns))
test = test.drop(columns_to_drop, axis=1).merge(temp, how='left', left_index=True, right_index=True)
del temp


# Feature aggregates

le_features = fa_features

categorial_features_encoders = encode_categorial_features_fit(
    pd.concat([train_fa, test_fa], join='outer', sort=False), le_features)

temp = encode_categorial_features_transform(train_fa, categorial_features_encoders)
columns_to_drop = list(set(le_features) & set(train_fa.columns))
train_fa = train_fa.drop(columns_to_drop, axis=1).merge(temp, how='left', left_index=True, right_index=True)
del temp

temp = encode_categorial_features_transform(test_fa, categorial_features_encoders)
columns_to_drop = list(set(le_features) & set(test_fa.columns))
test_fa = test_fa.drop(columns_to_drop, axis=1).merge(temp, how='left', left_index=True, right_index=True)
del temp


def make_aggregates(df, feature_to_group_by, feature):
    out = pd.DataFrame(index=df.index)
    agg = df.groupby([feature_to_group_by])[feature].value_counts(normalize=True)
    freq = lambda row: agg.loc[row[feature_to_group_by], row[feature]]
    out[feature + '_' + feature_to_group_by + '_freq'] = df.apply(freq, axis=1)
    return out


for feature in ['nom_5__bin_0', 'nom_6__bin_1']:
    feature_1, feature_2 = feature.split('__')
    print('Add feature:', feature, '/ aggregates of', feature_1, 'by', feature_2)
    
    agg = make_aggregates(train_fa, feature_2, feature_1)
    train = train.merge(agg, how='left', left_index=True, right_index=True)
    del agg
    
    agg = make_aggregates(test_fa, feature_2, feature_1)
    test = test.merge(agg, how='left', left_index=True, right_index=True)
    del agg

del train_fa
del test_fa


# Free memory

# From https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
#        else:
#            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


# Extract target variable

y_train = train['target'].copy()
x_train = train.drop('target', axis=1)
del train

x_test = test.copy()
del test


# LightGBM

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# Parameters from https://www.kaggle.com/pavelvpster/cat-in-dat-le-ohe-te-fe-lgb-bo

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'is_unbalance': False,
    'boost_from_average': True,
    'num_threads': 4,
    
    'num_iterations': 10000,
    'learning_rate': 0.006,
    'early_stopping_round': 100,
    
    'num_leaves': 94,
    'min_data_in_leaf': 61,
    'max_depth': 31,
    'bagging_fraction' : 0.12033530139527615,
    'feature_fraction' : 0.18631314159464357,
    'lambda_l1': 0.0628583713734685,
    'lambda_l2': 1.2728208225275608
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof, trained_estimators = fit(LightGBM(params), roc_auc_score, x_train.values, y_train.values, cv)
y = predict(trained_estimators, x_test.values)


# Submit predictions

submission = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv', index_col='id')
submission['target'] = y
submission.to_csv('lightgbm.csv')
"""
Beating the Benchmark 
Caterpillar @ Kaggle
Adapted from Abhishek's beating the benchmark v1.0
__author__ : Andrew Matteson

"""

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
import xgboost as xgb

# load training and test datasets
train = pd.read_csv('../input/train_set.csv', parse_dates=[2,])
test = pd.read_csv('../input/test_set.csv', parse_dates=[3,])

tubes = pd.read_csv('../input/tube.csv')

# create some new features
train['year'] = train.quote_date.dt.year
train['month'] = train.quote_date.dt.month
train['dayofyear'] = train.quote_date.dt.dayofyear
train['dayofweek'] = train.quote_date.dt.dayofweek
train['day'] = train.quote_date.dt.day

test['year'] = test.quote_date.dt.year
test['month'] = test.quote_date.dt.month
test['dayofyear'] = test.quote_date.dt.dayofyear
test['dayofweek'] = test.quote_date.dt.dayofweek
test['day'] = test.quote_date.dt.day

train = pd.merge(train,tubes,on='tube_assembly_id',how='inner')
test = pd.merge(test,tubes,on='tube_assembly_id',how='inner')

train['material_id'].fillna('SP-9999',inplace=True)
test['material_id'].fillna('SP-9999',inplace=True)

# drop useless columns and create labels
idx = test.id.values.astype(int)
test = test.drop(['id', 'tube_assembly_id', 'quote_date'], axis = 1)
labels = train.cost.values
train = train.drop(['quote_date', 'cost', 'tube_assembly_id'], axis = 1)

print (len(train.columns.tolist()))

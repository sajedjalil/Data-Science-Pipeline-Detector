"""
Beating the Benchmark 
Caterpillar @ Kaggle

__author__ : Abhishek

Change : log transformation of some of the predictors

"""

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
import xgboost as xgb

# load training and test datasets
train = pd.read_csv('../input/train_set.csv', parse_dates=[2,])
test = pd.read_csv('../input/test_set.csv', parse_dates=[3,])

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

# drop useless columns and create labels
idx = test.id.values.astype(int)
test = test.drop(['id', 'tube_assembly_id', 'quote_date'], axis = 1)
labels = train.cost.values
train = train.drop(['quote_date', 'cost', 'tube_assembly_id'], axis = 1)

train.quantity = np.log(1+train.quantity)
test.quantity = np.log(1+test.quantity)

train.annual_usage = np.log(1+train.annual_usage)
test.annual_usage = np.log(1+test.annual_usage)

train.min_order_quantity = np.log(1+train.min_order_quantity)
test.min_order_quantity = np.log(1+test.min_order_quantity)

print(train.head())

# convert data to numpy array
train = np.array(train)
test = np.array(test)

# label encode the categorical variables
for i in range(train.shape[1]):
    if i in [0,3]:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[:,i]) + list(test[:,i]))
        train[:,i] = lbl.transform(train[:,i])
        test[:,i] = lbl.transform(test[:,i])


# object array to float
train = train.astype(float)
test = test.astype(float)

# i like to train on log(1+x) for RMSLE ;) 
# The choice is yours :)
label_log = np.log1p(labels)

# fit a random forest model

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.1
params["min_child_weight"] = 5
params["subsample"] = 1.0
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 7

plst = list(params.items())

xgtrain = xgb.DMatrix(train, label=label_log)
xgtest = xgb.DMatrix(test)

num_rounds = 120
model = xgb.train(plst, xgtrain, num_rounds)

# get predictions from the model, convert them and dump them!
preds = np.expm1(model.predict(xgtest))
preds = pd.DataFrame({"id": idx, "cost": preds})
preds.to_csv('benchmark.csv', index=False)

# Swipe right on tinder ;)
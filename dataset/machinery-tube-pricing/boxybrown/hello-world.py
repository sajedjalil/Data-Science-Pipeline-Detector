
#######
## https://www.kaggle.com/andrewmatteson/caterpillar-tube-pricing/beating-the-benchmark-v1-0/run/20747
#######


import os
import csv
import pandas as pd
import numpy as np
import datetime
from sklearn import ensemble, preprocessing
import xgboost as xgb

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

### grab that data
train = pd.read_csv("../input/train_set.csv", parse_dates=[2,])
test = pd.read_csv("../input/test_set.csv", parse_dates=[3,])

tubes = pd.read_csv('../input/tube.csv')
tube_end = pd.read_csv('../input/tube_end_form.csv')

train = pd.merge(train,tubes,on='tube_assembly_id',how='inner')
test = pd.merge(test,tubes,on='tube_assembly_id',how='inner')

train = pd.merge(train,tube_end, left_on='end_a', right_on = 'end_form_id',how='left')
test = pd.merge(test,tube_end,left_on='end_a', right_on = 'end_form_id',how='left')

train['material_id'].fillna('SP-9999',inplace=True)
test['material_id'].fillna('SP-9999',inplace=True)

train['forming'].fillna('unknown', inplace = True)
test['forming'].fillna('unknown', inplace = True)

train['end_form_id'].fillna('unknown', inplace = True)
test['end_form_id'].fillna('unknown', inplace = True)

# create some new features
idx = test.id.values.astype(int)
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

# convert data to numpy array
train = np.array(train)
test = np.array(test)

# label encode the categorical variables
for i in range(train.shape[1]):
    if i in [0,3,5, 11,12,13,14,15,16,20, 21]:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[:,i]) + list(test[:,i]))
        train[:,i] = lbl.transform(train[:,i])
        test[:,i] = lbl.transform(test[:,i])


# object array to float
train = train.astype(float)
test = test.astype(float)

# i like to train on log(1+x) for RMSLE ;) 
label_log = np.log1p(labels)

# fit a random forest model

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.1
params["min_child_weight"] = 3
params["subsample"] = 0.8
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 8

plst = list(params.items())

xgtrain = xgb.DMatrix(train, label=label_log)
xgtest = xgb.DMatrix(test)

num_rounds = 120
model = xgb.train(plst, xgtrain, num_rounds)

# get predictions from the model, convert them and dump them!
preds = np.expm1(model.predict(xgtest))
preds = pd.DataFrame({"id": idx, "cost": preds})
preds.to_csv('benchmark.csv', index=False)





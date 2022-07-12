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

train = pd.merge(train,tubes,on='tube_assembly_id',how='inner')
test = pd.merge(test,tubes,on='tube_assembly_id',how='inner')

train['material_id'].fillna('SP-9999',inplace=True)
test['material_id'].fillna('SP-9999',inplace=True)

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
    print(i)
    if i in [0,3,5, 11,12,13,14,15,16]:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[:,i]) + list(test[:,i]))
        train[:,i] = lbl.transform(train[:,i])
        test[:,i] = lbl.transform(test[:,i])

print(train[0:5,:])

print(test[0:5,:])

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
params["min_child_weight"] = 3
params["subsample"] = .8
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 8

plst = list(params.items())

xgtrain = xgb.DMatrix(train, label=label_log)
xgtest = xgb.DMatrix(test)

num_rounds = 1000
model = xgb.train(plst, xgtrain, num_rounds)

# get predictions from the model, convert them and dump them!
preds = np.expm1(model.predict(xgtest))
preds = pd.DataFrame({"id": idx, "cost": preds})
preds.to_csv('benchmark.csv', index=False)

# Swipe right on grindr ;)
"""
Beating the Benchmark 
Caterpillar @ Kaggle

__author__ : pakozm

forked from Abhishek version

"""

import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import ensemble, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import ensemble

NUM_TREES=1000
MAX_FEATS="sqrt"

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
def fit(train_feats, train_labels):
    model = ensemble.RandomForestRegressor(n_estimators=NUM_TREES,
                                           max_features=MAX_FEATS,
                                           random_state=123,
                                           verbose=100,
                                           criterion="mse")
    model.fit(train_feats, train_labels)
    return model

model = fit(train, label_log)
train_p = model.predict(train)
print ("# TR LOSS",np.sqrt(sk.metrics.mean_squared_error(train_p, label_log)))

# get predictions from the model, convert them and dump them!
test_p = np.expm1(model.predict(test))

preds = pd.DataFrame({"cost": test_p}, index = idx)
preds.to_csv('benchmark.csv', index=True, index_label="id")

# Swipe right on tinder ;)

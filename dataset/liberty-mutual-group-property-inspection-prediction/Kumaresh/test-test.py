'''
This benchmark uses xgboost and early stopping to achieve a score of 0.38019
In the liberty mutual group: property inspection challenge

Based on Abhishek Catapillar benchmark
https://www.kaggle.com/abhishek/caterpillar-tube-pricing/beating-the-benchmark-v1-0

@author Devin

Have fun;)
'''

import pandas as pd
import numpy as np 
from sklearn import preprocessing, cross_validation
import xgboost as xgb
import random as random

#load train and test 
train  = pd.read_csv('../input/train.csv', index_col=0)
test  = pd.read_csv('../input/test.csv', index_col=0)

labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)

train.drop('T2_V10', axis=1, inplace=True)
train.drop('T2_V7', axis=1, inplace=True)
train.drop('T1_V13', axis=1, inplace=True)
train.drop('T1_V10', axis=1, inplace=True)

test.drop('T2_V10', axis=1, inplace=True)
test.drop('T2_V7', axis=1, inplace=True)
test.drop('T1_V13', axis=1, inplace=True)
test.drop('T1_V10', axis=1, inplace=True)


columns = train.columns
test_ind = test.index

train = np.array(train)
test = np.array(test)

# label encode the categorical variables
for i in range(train.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[:,i]) + list(test[:,i]))
    train[:,i] = lbl.transform(train[:,i])
    test[:,i] = lbl.transform(test[:,i])

train = train.astype(float)
test = test.astype(float)

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.005
params["min_child_weight"] =8
params["subsample"] = 0.85
params["colsample_bytree"] = 0.8
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 7

plst = list(params.items())

#Using 5000 rows for early stopping. 
offset = 5000

num_rounds = 2000
xgtest = xgb.DMatrix(test)

#create a train and validation dmatrices 
#xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
#xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

#Pop=range(0,50000)
#K1=random.sample(Pop,5000)
#K2=np.delete(Pop,K1)
#K2=Pop[~K1]


X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    train, labels, test_size=0.1, random_state=0)

xgtrain = xgb.DMatrix(X_train, label=y_train)
xgval = xgb.DMatrix(X_test, label=y_test)

#xgtrain = xgb.DMatrix(train[K2,:], label=labels[K2])
#xgval = xgb.DMatrix(train[K1,:], label=labels[K1])


#train using early stopping and predict
labels = np.log(labels)

watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
preds1 = model.predict(xgtest)
preds=preds1

for i in range(4):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        train, labels, test_size=0.1, random_state=(i+1))
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    xgval = xgb.DMatrix(X_test, label=y_test)
    #train using early stopping and predict
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=80)
    preds2 = model.predict(xgtest)
    preds=preds+(preds2)


train = train.T.to_dict().values()
test = test.T.to_dict().values()

vec = DictVectorizer()
train = vec.fit_transform(train)
test = vec.transform(test)


for i in range(4):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        train, labels, test_size=0.1, random_state=(i+1))
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    xgval = xgb.DMatrix(X_test, label=y_test)
    #train using early stopping and predict
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=80)
    preds2 = model.predict(xgtest)
    preds=preds+(preds2)

#generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('xgboost_benchmark.csv')
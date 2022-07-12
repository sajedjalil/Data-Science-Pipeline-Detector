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
from sklearn import preprocessing
import xgboost as xgb

#load train and test 
train  = pd.read_csv('../input/train.csv', index_col=0)
test  = pd.read_csv('../input/test.csv', index_col=0)

labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)


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
params["eta"] = 0.01
params["min_child_weight"] = 5
params["subsample"] = 0.86
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
xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

#train using early stopping and predict
watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=4)
preds1 = model.predict(xgtest)

#reverse train and labels and use different 5k for early stopping. 
# this adds very little to the score but it is an option if you are concerned about using all the data. 
train = train[::-1,:]
labels = np.log(labels[::-1])

xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=4)
preds2 = model.predict(xgtest)

model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=4)
preds3 = model.predict(xgtest)

model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=4)
preds4 = model.predict(xgtest)

model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=4)
preds5 = model.predict(xgtest)

model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=4)
preds6 = model.predict(xgtest)

model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=4)
preds7 = model.predict(xgtest)
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=4)
preds8 = model.predict(xgtest)


#combine predictions
#since the metric only cares about relative rank we don't need to average
preds = preds1 + preds2+preds3+preds4+preds5+preds6+preds7+preds8

#generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('xgboost_benchmark.csv')
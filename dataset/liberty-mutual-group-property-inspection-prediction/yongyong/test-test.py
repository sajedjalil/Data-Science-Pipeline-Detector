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
import random as random

#load train and test 
train  = pd.read_csv('../input/train.csv', index_col=0)
test  = pd.read_csv('../input/test.csv', index_col=0)

labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)

columns = train.columns
test_ind = test.index

print (train.shape, test.shape, list(columns))

train = np.array(train)
test = np.array(test)

# label encode the categorical variables
for i in range(train.shape[1]):
    #if ((i != 0) & (i != 1) & (i != 2) & (i != 9) & (i != 12) & (i != 13) & (i != 17) & (i != 18) & (i != 20) & (i != 22) & (i != 23) & (i != 24) & (i != 25) & (i != 26) & (i != 30) & (i != 31)) :
    print(i) 
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[:,i]) + list(test[:,i]))
    train[:,i] = lbl.transform(train[:,i])
    test[:,i] = lbl.transform(test[:,i])

train = train.astype(float)
test = test.astype(float)

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.05
params["min_child_weight"] =1
params["subsample"] = 0.5
params["colsample_bytree"] = 0.8
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 5

plst = list(params.items())

xgtrain = xgb.DMatrix(train, label=labels)
xgtest = xgb.DMatrix(test)

num_rounds = 1000
model = xgb.train(plst, xgtrain, num_rounds)

# get predictions from the model, convert them and dump them!
preds = model.predict(xgtest)

#generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('xgboost_benchmark.csv')





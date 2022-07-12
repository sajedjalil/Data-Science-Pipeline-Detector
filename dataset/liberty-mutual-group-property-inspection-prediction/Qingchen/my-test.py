'''


Based on Abhishek Catapillar benchmark
https://www.kaggle.com/abhishek/caterpillar-tube-pricing/beating-the-benchmark-v1-0

@author Devin

Have fun;)
'''

import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer


def xgboost_pred(train,labels,test):
    params = {}
    params["objective"] = "reg:linear"
    params["eval_metric"] = "rmse"
    params["eta"] = 0.005
    params["min_child_weight"] = 6
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.7
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 9

    plst = list(params.items())

    #Using 5000 rows for early stopping. 
    offset = 4000

    num_rounds = 10000
    xgtest = xgb.DMatrix(test)

    #create a train and validation dmatrices 
    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

    #train using early stopping and predict
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
    preds1 = model.predict(xgtest, ntree_limit=model.best_iteration)
    #preds1 = model.predict(xgtest)


    #reverse train and labels and use different 5k for early stopping. 
    # this adds very little to the score but it is an option if you are concerned about using all the data. 
    train = train[::-1,:]
    labels = np.log(labels[::-1])
    #labels = labels[::-1]

    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
    preds2 = model.predict(xgtest, ntree_limit=model.best_iteration)
    #preds2 = model.predict(xgtest)

    #combine predictions
    #since the metric only cares about relative rank we don't need to average
    preds = preds1*1.4 + preds2*8.6
    return preds

#load train and test 
train  = pd.read_csv('../input/train.csv', index_col=0)
test  = pd.read_csv('../input/test.csv', index_col=0)

labels = train.Hazard

train_s = train
test_s = test

train_s.drop('T2_V10', axis=1, inplace=True)
train_s.drop('T2_V7', axis=1, inplace=True)
train_s.drop('T1_V13', axis=1, inplace=True)
train_s.drop('T1_V10', axis=1, inplace=True)

test_s.drop('T2_V10', axis=1, inplace=True)
test_s.drop('T2_V7', axis=1, inplace=True)
test_s.drop('T1_V13', axis=1, inplace=True)
test_s.drop('T1_V10', axis=1, inplace=True)

columns = train.columns
test_ind = test.index

train_s = np.array(train_s)
test_s = np.array(test_s)

for i in range(train_s.shape[1]):
    if type(train_s[1, i]) is str:
        dic = {}
        for v in train_s[:, i]:
            if v not in dic.keys():
                col = train_s[train_s[:, i] == v]
                dic[v] = np.mean(col[:, 0])
        for n in range(0, len(train_s)):
            train_s[n, i] = dic[train_s[n, i]]
        for n in range(0, len(test_s)):
            test_s[n, i-1] = dic[test_s[n, i-1]]

train_s = np.column_stack([train_s, np.multiply(train_s[:, 1], train_s[:, 4])])
test_s = np.column_stack([test_s, np.multiply(test_s[:, 0], test_s[:, 3])])
train_s = np.column_stack([train_s, np.multiply(train_s[:, 1], train_s[:, 8])])
test_s = np.column_stack([test_s, np.multiply(test_s[:, 0], test_s[:, 7])])
train_s = np.column_stack([train_s, np.multiply(train_s[:, 1], train_s[:, 15])])
test_s = np.column_stack([test_s, np.multiply(test_s[:, 0], test_s[:, 14])])
train_s = np.column_stack([train_s, np.multiply(train_s[:, 1], train_s[:, 25])])
test_s = np.column_stack([test_s, np.multiply(test_s[:, 0], test_s[:, 24])]) 
train_s = np.column_stack([train_s, np.multiply(train_s[:, 3], train_s[:, 25])])
test_s = np.column_stack([test_s, np.multiply(test_s[:, 2], test_s[:, 24])])  
train_s = np.column_stack([train_s, np.multiply(train_s[:, 10], train_s[:, 13])])
test_s = np.column_stack([test_s, np.multiply(test_s[:, 9], test_s[:, 12])])  
train_s = np.column_stack([train_s, np.multiply(train_s[:, 10], train_s[:, 24])])
test_s = np.column_stack([test_s, np.multiply(test_s[:, 9], test_s[:, 23])])                                     
train_s = np.column_stack([train_s, np.multiply(train_s[:, 21], train_s[:, 22])])
test_s = np.column_stack([test_s, np.multiply(test_s[:, 20], test_s[:, 21])])  
train_s = np.column_stack([train_s, np.multiply(train_s[:, 22], train_s[:, 26])])
test_s = np.column_stack([test_s, np.multiply(test_s[:, 21], test_s[:, 25])])  
train_s = np.column_stack([train_s, np.multiply(train_s[:, 24], train_s[:, 25])])
test_s = np.column_stack([test_s, np.multiply(test_s[:, 23], test_s[:, 24])]) 
        
train_s = train_s.astype(float)
test_s = test_s.astype(float)

preds1 = xgboost_pred(train_s[::, 1::],labels,test_s)

#generate solution
preds1 = pd.DataFrame({"Id": test_ind, "Hazard": preds1})
preds1 = preds1.set_index('Id')
preds1.to_csv('my_test.csv')
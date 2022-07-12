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

def gini(solution, submission):                                                 
    df = sorted(zip(solution, submission), key=lambda x : (x[1], x[0]),  reverse=True)
    random = [float(i+1)/float(len(df)) for i in range(len(df))]                
    totalPos = np.sum([x[0] for x in df])                                       
    cumPosFound = np.cumsum([x[0] for x in df])                                     
    Lorentz = [float(x)/totalPos for x in cumPosFound]                          
    Gini = [l - r for l, r in zip(Lorentz, random)]                             
    return np.sum(Gini)                                                         


def normalized_gini(solution, submission):                                      
    normalized_gini = gini(solution, submission)/gini(solution, solution)       
    return normalized_gini        

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
params["eta"] = 0.01
params["min_child_weight"] = 5
params["subsample"] = 0.8
params["colsample_bytree"] = 0.8
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 8

plst = list(params.items())

#Using 5000 rows for early stopping. 
offset = 5000

num_rounds = 20000
xgtest = xgb.DMatrix(test)

n = train.shape[0]
build_ind = np.ones(n, dtype=bool)
build_ind[offset:offset*2] = False
valid_ind = np.zeros(n, dtype=bool)
valid_ind[offset:offset*2] = True

#create a train and validation dmatrices 
xgtrain = xgb.DMatrix(train[build_ind,:], label=labels[build_ind])
xgval = xgb.DMatrix(train[valid_ind,:], label=labels[valid_ind])

#train using early stopping and predict
watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
preds1 = model.predict(xgtest)
print(normalized_gini(solution=labels[valid_ind], submission=model.predict(xgval)))

preds = preds1

#generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('_xgboost_benchmark_1b.csv')
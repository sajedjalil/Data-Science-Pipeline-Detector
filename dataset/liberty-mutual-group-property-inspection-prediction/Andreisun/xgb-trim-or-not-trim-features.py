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
    df = zip(solution, submission)                                                                                                                                                                                                             
    df = sorted(df, key=lambda x: (x[1],x[0]), reverse=True)                                                                                                                                                                                   
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]                                                                                                                                                                                 
    totalPos = float(sum([x[0] for x in df]))                                                                                                                                                                                                  
    cumPosFound = [df[0][0]]                                                                                                                                                                                                                   
    for i in range(1,len(df)):                                                                                                                                                                                                                 
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])                                                                                                                                                                         
    Lorentz = [float(x)/totalPos for x in cumPosFound]                                                                                                                                                                                         
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]                                                                                                                                                                                        
    return sum(Gini)                                                                                                                                                                                                                           
                                                                                                                                                                                                                                               
def normalized_gini(solution, submission):                                                                                                                                                                                                     
    normalized_gini = gini(solution, submission)/gini(solution, solution)                                                                                                                                                                      
    return normalized_gini  
    

#load train and test 
train  = pd.read_csv('../input/train.csv', index_col=0)
test  = pd.read_csv('../input/test.csv', index_col=0)

labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)

cols = ['T2_V10', 'T2_V7', 'T1_V13', 'T1_V10']

train.drop( cols, axis = 1, inplace=True )
test.drop( cols, axis = 1, inplace=True )

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
offset = 10000 #train.shape[0] / 2 -1

num_rounds = 2000

xgtest = xgb.DMatrix(test)

#scores =[]

#create a train and validation dmatrices 
xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

#train using early stopping and predict
watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
pred_val = model.predict(xgb.DMatrix(train[:offset]))
preds1 = model.predict(xgtest)

model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
pred_val += model.predict(xgb.DMatrix(train[:offset]))
preds1 += model.predict(xgtest)

model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
pred_val += model.predict(xgb.DMatrix(train[:offset]))
preds1 += model.predict(xgtest)

#score = normalized_gini(labels[:offset], pred_val)
#scores.append(score)

#reverse train and labels and use different 5k for early stopping. 
# this adds very little to the score but it is an option if you are concerned about using all the data. 
train = train[::-1,:]
labels = labels[::-1]
#labels = np.log(labels[::-1])

xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
pred_val = model.predict(xgb.DMatrix(train[:offset]))
preds2 = model.predict(xgtest)

model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
pred_val += model.predict(xgb.DMatrix(train[:offset]))
preds3 = model.predict(xgtest)

model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
pred_val += model.predict(xgb.DMatrix(train[:offset]))
preds4 = model.predict(xgtest)

model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
pred_val += model.predict(xgb.DMatrix(train[:offset]))
preds5 = model.predict(xgtest)

model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
pred_val += model.predict(xgb.DMatrix(train[:offset]))
preds6 = model.predict(xgtest)

model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
pred_val += model.predict(xgb.DMatrix(train[:offset]))
preds7 = model.predict(xgtest)

model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
pred_val += model.predict(xgb.DMatrix(train[:offset]))
preds8 = model.predict(xgtest)

score = normalized_gini(labels[:offset], pred_val)
print(score)

#combine predictions
#since the metric only cares about relative rank we don't need to average
preds = preds1 + preds2 + preds3 + preds4 + preds5 + preds6 + preds7 + preds8

#generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('xgboost_benchmark.csv')

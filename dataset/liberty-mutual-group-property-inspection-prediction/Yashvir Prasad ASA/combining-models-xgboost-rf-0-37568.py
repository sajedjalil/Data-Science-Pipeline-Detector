# Not a script you just run!!

# This script shows the value of combining various models
# This script combines the xgboost authored by Devin and the RF authored by juandoso
# To make this run in the necessary time, I reduced the time the xgboost and rf ran
# If you increase the time, you will get better results
# Please comment how well this works 

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv", index_col=0)
test  = pd.read_csv("../input/test.csv", index_col=0 )

labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)
train.drop('T1_V10', axis=1, inplace=True)
train.drop('T1_V13', axis=1, inplace=True)
train.drop('T2_V7', axis=1, inplace=True)
train.drop('T2_V10', axis=1, inplace=True)

test.drop('T1_V10', axis=1, inplace=True)
test.drop('T1_V13', axis=1, inplace=True)
test.drop('T2_V7', axis=1, inplace=True)
test.drop('T2_V10', axis=1, inplace=True)

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

##RF Model 
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
#param_grid = {'n_estimators': [50, 100]} #original value
param_grid = {'n_estimators': [15, 25]}
model = GridSearchCV(RandomForestRegressor(), param_grid)
model = model.fit(train,labels)
print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)
predsrf = model.predict(test)

#xgboost
params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.01
params["min_child_weight"] = 5
params["subsample"] = 0.8
params["colsample_bytree"] = 0.8
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 7

plst = list(params.items())

#Using 5000 rows for early stopping. 
offset = 5000

num_rounds = 100 #was 2000 in original script
xgtest = xgb.DMatrix(test)

#create a train and validation dmatrices 
xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

#train using early stopping and predict
watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=5)
preds1 = model.predict(xgtest)

#reverse train and labels and use different 5k for early stopping. 
# this adds very little to the score but it is an option if you are concerned about using all the data. 
train = train[::-1,:]
labels = labels[::-1]

xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=5)
preds2 = model.predict(xgtest)

#Combine Results
total = preds1 + preds2 + predsrf
preds = pd.DataFrame({"Id": test_ind, "Hazard": total})
preds = preds.set_index('Id')
preds.to_csv('xgboostandRF_benchmark.csv')
'''


Based on Abhishek Catapillar benchmark
https://www.kaggle.com/abhishek/caterpillar-tube-pricing/beating-the-benchmark-v1-0

@author Devin

Have fun;)
'''

import pandas as pd
import numpy as np 
from sklearn import ensemble,  preprocessing
import xgboost as xgb

#load train and test 
train  = pd.read_csv('../input/train.csv', index_col=0)
test  = pd.read_csv('../input/test.csv', index_col=0)

#train  = pd.read_csv('Data/train.csv', index_col=0)
#test  = pd.read_csv('Data/test.csv', index_col=0)

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

clf2 = ensemble.ExtraTreesRegressor(n_jobs=-1, n_estimators=100, random_state=0)
clf2.fit(train, labels)
preds_et1 = clf2.predict(test)

#clf2 = ensemble.GradientBoostingRegressor(n_estimators=200, random_state=0, max_depth=9, subsample=0.95)
#clf2.fit(train, labels)
#pred_gbr1 = clf2.predict(test)

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.005
params["min_child_weight"] = 5
params["subsample"] = 0.8
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
#model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=10)
#preds1 = model.predict(xgtest)

#model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=15)
#preds21 = model.predict(xgtest)

#model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=8)
#preds1 = preds1 + model.predict(xgtest)

#reverse train and labels and use different 5k for early stopping. 
# this adds very little to the score but it is an option if you are concerned about using all the data. 
train = train[::-1,:]
labels = np.log(labels[::-1])

#xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
#xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

#watchlist = [(xgtrain, 'train'),(xgval, 'val')]
#model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=6)
#preds2 = model.predict(xgtest)

#model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=4)
#preds3 = model.predict(xgtest)

#model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=10)
#preds4 = model.predict(xgtest)

#model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=15)
#preds5 = model.predict(xgtest)

#model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=8)
#preds6 = model.predict(xgtest)

clf2 = ensemble.ExtraTreesRegressor(n_jobs=-1, n_estimators=100, random_state=0)
clf2.fit(train, labels)
preds_et2 = clf2.predict(test)

#clf2 = ensemble.GradientBoostingRegressor(n_estimators=200, random_state=0, max_depth=9, subsample=0.95)
#clf2.fit(train, labels)
#pred_gbr2 = clf2.predict(test)

#model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=10)
#preds7 = model.predict(xgtest)

#model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=10)
#preds8 = model.predict(xgtest)


#combine predictions
#since the metric only cares about relative rank we don't need to average
#preds = (preds1+preds21)*2.8/3.0 #+ (preds2*2+preds3*1+preds4*4+preds5*3+preds6*1)*7.0/11.0 +0.4*(preds_et1*0.5+preds_et2*1)+0.4*(pred_gbr1*0.5+pred_gbr2*1.5)#+preds7+preds8



preds =0.4*(preds_et1*0.5+preds_et2*1)

#generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('xgboost_benchmark67_ss_ET0.4.csv')


#preds =0.4*(pred_gbr1*0.5+pred_gbr2*1.5)

#generate solution
#preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
#preds = preds.set_index('Id')
#preds.to_csv('xgboost_benchmark67_ss_gbr0.4.csv')


#preds.to_csv('xgboost_benchmark67_ss_2.8-3.0.csv')
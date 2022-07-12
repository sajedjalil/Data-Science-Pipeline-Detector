
import numpy as np
import xgboost as xgb
import csv
from random import randint
import json
from scipy.sparse import csr_matrix
from xgboost.sklearn import XGBClassifier
from sklearn import (metrics, cross_validation, linear_model, preprocessing)   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.preprocessing import StandardScaler # we will use this to standardize the data
from sklearn.metrics import roc_auc_score # the metric we will be tested on . You can find more here :  https://www.kaggle.com/wiki/AreaUnderCurve
from sklearn.cross_validation import StratifiedKFold # the cross validation method we are going to use
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from math import log
import pandas as pd
# function for save csv result
def save_result(result,ID,preds):
	print ("save result")
	with open(result, 'w') as fp:
		a = csv.writer(fp, delimiter=',')
		a.writerow(["ID", "TARGET" ])
		data = zip(ID,preds)
		a.writerows(data)


print ("loading data")
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')

## remove IDS
ID = test['ID']
test.drop('ID', axis=1, inplace=True)
train.drop('ID', axis=1, inplace=True)


##### Extracting TARGET
y =  train['TARGET']
train.drop('TARGET', axis=1, inplace=True)


var15 = test['var15']
saldo_medio_var5_hace2 = test['saldo_medio_var5_hace2']
saldo_var33 = test['saldo_var33']
var38 = test['var38']
v21 = test['var21']
nv = test['num_var33']+test['saldo_medio_var33_ult3']+test['saldo_medio_var44_hace2']+test['saldo_medio_var44_hace3']+test['saldo_medio_var33_ult1']+test['saldo_medio_var44_ult1']
num_var30 = test['num_var30']
num_var13_0 = test['num_var13_0']
num_var33_0 = test['num_var33_0']


#add number of zero feature

print ("preprocessing data")
features = train.columns
train.insert(1, 'SumZeros', (train[features] == 0).astype(int).sum(axis=1))
features = test.columns
test.insert(1, 'SumZeros', (test[features] == 0).astype(int).sum(axis=1))



##### Removing constant features
remove = []
c = train.columns
print ("Remove constant features")
for i in range(len(c)-1):
    v = train[c[i]].values
    if len(set(v)) == 1:
    	remove.append(c[i])

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)



# remove identical features
remove = []
c = train.columns
for i in range(len(c)-1):
    v = train[c[i]].values
    for j in range(i+1, len(c)):
    	if np.array_equal(v, train[c[j]].values):
    		remove.append(c[j])


train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)


train['var38'] = np.log10(np.abs((train['var38'].values)))
test['var38']  = np.log10(np.abs((test['var38'].values))) 

#---limit vars in test based on min and max vals of train
print('Setting min-max lims on test data')
c = train.columns
for i in range(len(c)-1):
	lim = min(train[c[i]].values)
	test[c[i]][test[c[i]].values < lim] = lim
	lim = max(train[c[i]].values)
	test[c[i]][test[c[i]].values > lim] = lim 

deep_learn_train = train
deep_learn_test = test
deep_learn_train['TARGET'] = y
deep_learn_train.insert(0, 'ID', ID)
deep_learn_test.insert(0, 'ID', ID)

deep_learn_train.to_csv('deep_learn_train.csv')
deep_learn_test.to_csv('deep_learn_test.csv')

params = {}
params["objective"] = "binary:logistic"
params["eta"] = 0.0202048
params["subsample"] = 0.6815
params["colsample_bytree"] = 0.701
params["max_depth"] = 5
params["booster"] = 'gbtree'
params["eval_metric"] = "auc"
params['seed'] = 1234


dtrain = xgb.DMatrix(csr_matrix(train), y,silent=True)
watchlist = [(dtrain, 'train')]
clf =  xgb.train(   params,  dtrain, 560,  evals=watchlist , verbose_eval=False)


#######actual variables

dtest = xgb.DMatrix(csr_matrix(test), silent=True)
preds  =  clf.predict(dtest)
pred  = clf.predict(dtrain)



preds[var15 < 23] = 0
preds[saldo_medio_var5_hace2 > 160000] = 0
preds[saldo_var33 > 0] = 0
preds[var38 > 3988596] = 0
preds[nv > 0] = 0
preds[v21 > 7500] = 0
preds[num_var30 > 9] = 0
preds[num_var13_0 > 6] = 0
preds[num_var33_0 > 0] = 0
print('Average ROC:', roc_auc_score(y, pred))


save_result("xgb.csv", ID, preds)
print ("start cross validation")


import time
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import svm, cross_validation, preprocessing
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.metrics import accuracy_score, log_loss

#############################################################################################
# Import data, Preprocess
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
labels = train[['target']].values
s = {'Class_1':1, 'Class_2':2, 'Class_3':3, 'Class_4':4, 'Class_5':5, 'Class_6':6,'Class_7':7, 'Class_8':8, 'Class_9':9}
labels = [s[k] for k in labels[:,0]]

train.drop(['id','target'], axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

perms = np.random.permutation(train.shape[0])

train = np.array(train)[perms,:]
labels = np.array(labels)[perms]

xtr, xte, xtest = np.array(train)[0:60000,:], np.array(train)[60001:61877,], np.array(test)
ytr, yte = np.array(labels)[0:60000,], np.array(labels)[60001:61877,]

print ('I. Data Imported, preprocessed')

#############################################################################################
# Gradient Boosting
#gbc2 = GBC(n_estimators = 50, max_depth = 5, verbose = 1, learning_rate = 0.1)
#gbc2.fit(xtr, ytr)

#print ('II. Trained: gradient boosting classifier')
#print (log_loss(yte, gbc2.predict_proba(xte)))

#############################################################################################
# Extreme Gradient Boosting
xgbxtr = xgb.DMatrix(xtr, label = ytr-1)
xgbxte = xgb.DMatrix(xte, label = yte-1)

param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.3
param['max_depth'] = 10
param['eval_metrix'] = 'mlogloss'
param['num_class'] = 9
param['nthread'] = 8
param['verbose'] = 1

watchlist = [(xgbxtr, 'train'),(xgbxte, 'test')]
nround = 75
bst3 = xgb.train(param, xgbxtr, nround, watchlist);

print ('III. Trained: extreme gradient boosting classifier')
print (log_loss(yte, bst3.predict(xgbxte)))

##############################################################################################
# Extra Trees Classifier
etc4 = ETC(n_estimators = 100, max_depth = 40, max_features = 15)
stme = time.time()
etc4.fit(xtr, ytr)
print (time.time() - stme)
print ('IV. Trained: extra trees classifier')
print (log_loss(yte, etc4.predict_proba(xte)))

##############################################################################################
# Weigh the ensemble
predMin = etc4.predict_proba(xte)
predMax = etc4.predict_proba(xte)
predxgb = bst3.predict(xgb.DMatrix(xte))

w1, w2, w3 = 0.1, 0.0, 0.9
preds2 = (w1*predMin + w2*predMax + w3*predxgb)

print ('Log-loss (test): ')
print (log_loss(yte, preds2))




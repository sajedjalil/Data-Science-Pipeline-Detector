import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation
import numpy as np
import gc
from sklearn import metrics

print("reading the train and test data\n")
train = pd.read_csv("../input/train.csv")

target = train.target.values
train.drop(['ID','target'], axis=1, inplace=True)

print("assuming text variables are categorical & replacing them with numeric ids\n")
for c in train.columns: 
    train[c].fillna(value=-1.0, inplace=True)
    le = LabelEncoder()
    train[c] = le.fit_transform(train[c].astype(str))
        
  
gc.collect()

kf = cross_validation.StratifiedKFold(target, n_folds=3, shuffle=True, random_state=42)
for a,b in kf:
    train = train.loc[a,:]; target = target[a].ravel()
    test = train.loc[b,:]; targetTest = target[b].ravel
    break

print("training a XGBoost classifier\n")
dtrain = xgb.DMatrix(train.values, label=target)
dtest = xgb.DMatrix(test.values, label=targetTest)

param = {'max_depth':2, 
         'eta':1, 
         'objective':'binary:logistic', 
         'eval_metric': 'auc',
         'verbose':2}
clf = xgboost.train(param, dtrain, 20)

preds = clf.predict(dtest)

print(metrics.roc_auc_score(targetTest, preds)   )

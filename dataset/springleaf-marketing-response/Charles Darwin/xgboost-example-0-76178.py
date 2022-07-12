import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import gc

print("reading the train and test data\n")
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

featureNames = train.columns[1:-1]

print("assuming text variables are categorical & replacing them with numeric ids\n")
for c in featureNames: 
    if train[c].dtype.name == 'object':
        le = LabelEncoder()
        le.fit(np.append(train[c], test[c]))
        
        train[c] = le.transform(train[c]).astype(int)     
        test[c] = le.transform(test[c]).astype(int) 
  

print("replacing missing values with -1\n")
train = train.fillna(-1)
test = test.fillna(-1)
gc.collect()

print("sampling train to get around 8GB memory limitations\n")
train = train.sample(n=40000)
gc.collect()

print("training a XGBoost classifier\n")
dtrain = xgb.DMatrix(train[featureNames].values, label=train['target'].values)

param = {'max_depth':2, 
         'eta':1, 
         'objective':'binary:logistic', 
         'eval_metric': 'auc'}
clf = xgboost.train(param, dtrain, 20)


print("making predictions in batches due to 8GB memory limitation\n")
submission = test[['ID']]
submission[['target']] = np.nan
step = len(submission)/10000
for rows in xrange(0, len(submission), step):
    submission.loc[rows:rows+step, "target"] = clf.predict(xgb.DMatrix(test.loc[rows:rows+step, featureNames].values))


print("saving the submission file\n")
submission.to_csv("xgboost_submission.csv", index=False)

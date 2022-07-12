import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

#seed = 42

train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')



test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')

y = train.target.values
train = train.drop(['id', 'target'], axis=1)
test = test.drop('id', axis=1)

train = train.fillna(-1)
test = test.fillna(-1)

for f in train.columns:
    if train[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

clf = xgb.XGBClassifier(n_estimators=650,
                        nthread=-1,
                        max_depth=8,
                        learning_rate=0.026,
                        silent=True,
                        subsample=0.1,
                        colsample_bytree=0.7)
                        
xgb_model = clf.fit(train, y, eval_metric="auc")

preds = clf.predict_proba(test)[:,1]
sample = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv')

sample.target = preds
sample.to_csv('submission.csv', index=False)
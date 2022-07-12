# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_auc_score

training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)
print(test.shape)

X = training.iloc[:,:-1]
y = training.TARGET

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1301, stratify=y)
clf1 = xgb.XGBClassifier(max_depth=5, n_estimators=400, learning_rate=0.05)
clf1.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc",
        eval_set=[(X_test, y_test)])
        
print('XG1 test AUC:', roc_auc_score(y_test, clf1.predict_proba(X_test)[:, 1]))
        
y_pred1 = clf1.predict_proba(X)
print('XG1 AUC:', roc_auc_score(y, y_pred1[:,1]))

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
clf2 = xgb.XGBClassifier(max_depth=5, n_estimators=500, learning_rate=0.05)
clf2.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc",
        eval_set=[(X_test, y_test)])
        
print('XG2 test AUC:', roc_auc_score(y_test, clf2.predict_proba(X_test)[:, 1]))
        
y_pred2 = clf2.predict_proba(X)
print('XG2 AUC:', roc_auc_score(y, y_pred2[:,1]))

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=2000, random_state=1301, n_jobs=-1, oob_score=True)
rfc.fit(X, y)
y_pred3=  rfc.predict_proba(X)
print('RF AUC:', roc_auc_score(y, y_pred3[:,1]))

y_pred = (y_pred1[:,1] + y_pred2[:,1] + y_pred3[:,1])/3

print('Overall AUC:', roc_auc_score(y, y_pred))

y_pred1 = clf1.predict_proba(test)
y_pred2 = clf2.predict_proba(test)
y_pred3 = rfc.predict_proba(test)

y_pred = (y_pred1[:,1] + y_pred2[:,1] + y_pred3[:,1])/3

submission = pd.DataFrame({"ID":test.index, "TARGET": y_pred})
submission.to_csv("submission.csv", index=False)

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# load data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# remove constant columns
remove = []
for col in df_train.columns:
    if df_train[col].std() == 0:
        remove.append(col)

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
c = df_train.columns
for i in range(len(c)-1):
    v = df_train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,df_train[c[j]].values):
            remove.append(c[j])

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID','TARGET'], axis=1).values

id_test = df_test['ID']
X_test = df_test.drop(['ID'], axis=1).values

# length of dataset
len_train = len(X_train)
len_test  = len(X_test)

# classifier
clf1 = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=200, learning_rate=0.05, nthread=-1, subsample=0.95, colsample_bytree=0.85, seed=4242)
#clf=RandomForestClassifier(n_jobs=-1, n_estimators=500)
clf2=GradientBoostingClassifier(loss='deviance', learning_rate=0.05, 
                n_estimators=500, subsample=1.0, min_samples_split=2, 
                min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                max_depth=5, init=None, random_state=None, max_features=None, 
                verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
                
X_fit, X_eval, y_fit, y_eval= train_test_split(X_train, y_train, test_size=0.3)

# fitting
#clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)

#print('Overall AUC:', roc_auc_score(y_train, (clf1.predict_proba(X_train)[:,1]+clf2.predict_proba(X_train)[:,1])/2)

# predicting
y_pred= clf1.predict_proba(X_test)[:,1]
y_pred=y_pred+ clf2.predict_proba(X_test)[:,1]
y_pred=y_pred/2
#y_pred=(y_preds+y_pred)/2

submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
submission.to_csv("submission.csv", index=False)

print('Completed!')
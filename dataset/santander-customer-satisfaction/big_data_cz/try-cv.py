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

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 14:46:21 2016

@author: yihongchen
"""

#python equivalent to Santander - Starter - parameter tune (from Fork) with Annealing

import time; start_time = time.time()
import numpy as np
import pandas as pd
from sklearn import ensemble
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import random; random.seed(2016)

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
num_train = train.shape[0]

d_col_drops =[]
c = train.columns
for i in range(len(c)-1):
    v = train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,train[c[j]].values):
            d_col_drops.append(c[i])
            print("----- dropping column: ", c[i], " ------ duplicate of: ", c[j], " ----------")
            break
train = train.drop(d_col_drops,axis=1)
test = test.drop(d_col_drops,axis=1)

y_train = train['TARGET']
train = train.drop(['TARGET'],axis=1)
id_test = test['ID']

df_all = pd.concat((train, test), axis=0, ignore_index=True)
df_all['null_count'] = df_all.isnull().sum(axis=1).tolist()
df_all['zero_count'] = df_all.apply( lambda x : x.value_counts().get(0,0), axis=1)
df_all_temp = df_all['ID']
df_all = df_all.drop(['ID'],axis=1)

df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]

g={'ne':352,'md':5,'mf': 1.0,'rs':2016}
xgc = xgb.XGBClassifier(n_estimators=g['ne'], max_depth=g['md'], seed=g['rs'], missing=np.nan, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7, objective='binary:logistic') 
model = xgc
model.fit(train, y_train.values, eval_metric="auc", eval_set=[(train, y_train.values)], verbose=10, early_stopping_rounds=20)
best_score = (roc_auc_score(y_train.values, model.predict_proba(train)[:,1]))
y_pred = model.predict_proba(test)[:,1]
print("Best CV score: ", best_score, " Time: ", round(((time.time() - start_time)/60),2))

for i in range(len(y_pred)):
    if y_pred[i]<0.0:
        y_pred[i] = 0.0
    if y_pred[i]>1.0:
        y_pred[i] = 1.0
pd.DataFrame({"ID": id_test, "TARGET": y_pred}).to_csv('submission.csv',index=False)
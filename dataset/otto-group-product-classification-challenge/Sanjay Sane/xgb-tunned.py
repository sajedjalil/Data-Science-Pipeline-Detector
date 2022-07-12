# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
os.chdir("/kaggle/input/otto-group-product-classification-challenge")

train = pd.read_csv("train.csv",index_col=0)
test  = pd.read_csv("test.csv",index_col=0)

#print(train["target"].unique())

X = train.iloc[:,:-1]
y = train.iloc[:,-1]


lr_range = np.linspace(0.001,1)
gamma_range = np.linspace(0.001,1500)
tm_range = ['hist','approx']
min_child_weight_range = np.arange(1,50,3)
colsample_bytree_range = np.linspace(0.1,0.7,5)
depth_range = np.arange(3,30,5)


parameters = dict(learning_rate=lr_range,gamma=gamma_range,tree_method=tm_range,
                  min_child_weight=min_child_weight_range,max_depth=depth_range,
                  colsample_bytree=colsample_bytree_range)


from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=42)

from sklearn.model_selection import RandomizedSearchCV
clf = XGBClassifier(random_state=2000)
rcv = RandomizedSearchCV(clf, param_distributions=parameters,
                  cv=kfold,scoring='neg_log_loss',n_iter=10,n_jobs=-1,
                  random_state=2020,verbose=4)

rcv.fit(X,y)

import pickle
pkfile = open('xgb.pkl', 'wb') 
pickle.dump(rcv, pkfile)   

best_model = rcv.best_estimator_


y_ans = best_model.predict_proba(test)


sampsub = pd.read_csv("sampleSubmission.csv")
submit = pd.DataFrame(y_ans,index=test.index,columns=sampsub.columns[1:]) 

submit.to_csv("submit_XGB.csv")
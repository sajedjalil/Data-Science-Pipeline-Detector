# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.metrics import roc_auc_score,roc_curve,make_scorer
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM,SVC,LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
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

'''
scaler = StandardScaler().fit(X_train)
X = scaler.transform(X_train)
unpredict = scaler.transform(X_test)


scaler=MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
unpredict = scaler.transform(X_test)    
'''

score=make_scorer(roc_auc_score)
X_fit, X_eval, y_fit, y_eval= train_test_split(X_train, y_train, test_size=0.3)
# classifier
clf=xgb.XGBClassifier(missing=np.nan, max_depth=4, n_estimators=500, learning_rate=0.03, nthread=4, subsample=0.95,
                        colsample_bytree=0.85,  seed=4242,objective='binary:logitraw')
#clf= CalibratedClassifierCV(model,method='isotonic',cv=3)                 
clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc",
                    eval_set=[(X_fit, y_fit), (X_eval, y_eval)])
print('train AUC:', roc_auc_score(y_fit, clf.predict_proba(X_fit)[:, 1]))
print('test AUC:', roc_auc_score(y_eval, clf.predict_proba(X_eval)[:, 1]))





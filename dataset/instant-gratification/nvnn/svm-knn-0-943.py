# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import os
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time
import gc
from sklearn import neighbors

from sklearn import metrics, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

X = train.iloc[:,1:257]
X_test = test.iloc[:,1:257]
Y = train.iloc[:,257]

cols = [c for c in train.columns if c not in ['id', 'target']]

cols.remove('wheezy-copper-turtle-magic')

prediction = np.zeros(len(test))

scaler = preprocessing.StandardScaler()
scaler.fit(pd.concat([X, X_test]))
scaler.fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)

###############################################################################
################################## Model
###############################################################################
skf = StratifiedKFold(n_splits=5, random_state=42)

oof = np.zeros(len(train))
st = time.time()
for i in range(512):
    if i%5==0: print('Model : ',i, 'Time : ', time.time()-st)

    x = train[train['wheezy-copper-turtle-magic']==i]
    x_test = test[test['wheezy-copper-turtle-magic']==i]
    y = Y[train['wheezy-copper-turtle-magic']==i]
    idx = x.index
    idx_test = x_test.index
    x.reset_index(drop=True,inplace=True)
    x_test.reset_index(drop=True,inplace=True)
    y.reset_index(drop=True,inplace=True)
    
    clf = lgb.LGBMRegressor()
    clf.fit(x[cols],y)
    important_features = [i for i in range(len(cols)) if clf.feature_importances_[i] > 0] 
    cols_important = [cols[i] for i in important_features]
    
    skf = StratifiedKFold(n_splits=10, random_state=42)
    for train_index, valid_index in skf.split(x.iloc[:,1:-1], y):
        # KNN
        clf = neighbors.KNeighborsClassifier(n_neighbors  =7, p=2, weights ='distance')
        clf.fit(x.loc[train_index][cols_important], y[train_index])
        oof[idx[valid_index]] = clf.predict_proba(x.loc[valid_index][cols_important])[:,1]
        prediction[idx_test] += clf.predict_proba(x_test[cols_important])[:,1] / 10.0
    print(i, 'oof auc : ', roc_auc_score(Y[idx], oof[idx]))
        

print('total auc : ',roc_auc_score(train['target'],oof))

#sub = pd.read_csv('data/sample_submission.csv')
#sub['target'] = prediction
#sub.to_csv('submission.csv',index=False)

import numpy as np, pandas as pd, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

#train = pd.read_csv('data/train.csv')
#test = pd.read_csv('data/test.csv')

oof = np.zeros(len(train))
preds = np.zeros(len(test))
cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

# BUILD 512 SEPARATE NON-LINEAR MODELS
for i in range(512):
    
    # EXTRACT SUBSET OF DATASET WHERE WHEEZY-MAGIC EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
    
    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
        
    # STRATIFIED K FOLD (Using splits=25 scores 0.002 better but is slower)
    skf = StratifiedKFold(n_splits=11, random_state=42)
    for train_index, test_index in skf.split(train3, train2['target']):
        
        # MODEL WITH SUPPORT VECTOR MACHINE
        clf = SVC(probability=True,kernel='poly',degree=4,gamma='auto')
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
        
    #if i%10==0: print(i)
        
# PRINT VALIDATION CV AUC
auc = roc_auc_score(train['target'],oof)
print('CV score =',round(auc,5))

sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds/2 + prediction/2
sub.to_csv('submission.csv',index=False)

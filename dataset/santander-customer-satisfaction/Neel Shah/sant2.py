import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pylab as plt
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

data=pd.read_csv('../input/train.csv',sep=",")
data = data.replace(-999999,2)
numCol=len(data.columns)
numRow=len(data)
target=[]
target=data["TARGET"]
finData=[]
id=[]
id=data["ID"]
finData=data.iloc[:,1:numCol-1]
numCol=len(finData.columns)
numRow=len(finData)

data_test=pd.read_csv('../input/test.csv',sep=",")
data_test = data_test.replace(-999999,2)
id_test=[]
id_test=data_test["ID"]
numColt=len(data_test.columns)
numRowt=len(data_test)
finDatat=[]
finDatat=data_test.iloc[:,1:numColt]
numColt=len(finDatat.columns)
numRowt=len(finDatat)

scaler=preprocessing.StandardScaler().fit(finData)
train_scaled=scaler.transform(finData)
test_scaled=scaler.transform(finDatat)

p=PCA(n_components=train_scaled.shape[1])
p.fit(train_scaled)
trainX=p.transform(train_scaled)
testX=p.transform(test_scaled)

t=p.explained_variance_
s = np.arange(1, 370, 1)

tr=trainX[:,0:142]
te=testX[:,0:142]
from sklearn import cross_validation

import xgboost as xgb

X_train, X_test, y_train, y_test = \
   cross_validation.train_test_split(tr, target, random_state=1301, stratify=target, test_size=0.35)
clf = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.85,
       gamma=0, learning_rate=0.02, max_delta_step=0, max_depth=3,
       min_child_weight=3, missing=None, n_estimators=441, nthread=4,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=4242, silent=True, subsample=0.95)

#clf.fit(tr, target, eval_metric="auc", eval_set=[(tr, target)])
from sklearn.metrics import roc_auc_score
####################
clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",
         eval_set=[(X_test, y_test)])
print('Overall AUC:', roc_auc_score(target, clf.predict_proba(tr)[:,1]))
y_pred = clf.predict_proba(X_test)
max=np.argmax(y_pred,axis=1)
lab = y_test.as_matrix()
correct_pred = 0
for i in range(len(max)):
    if max[i] == lab[i]:
        correct_pred += 1
print((100*correct_pred) / len(lab))

####################
print('Overall AUC on whole train set:', roc_auc_score(target, clf.predict_proba(tr)[:,1]))

clf.fit(tr, target, eval_metric="auc", eval_set=[(tr, target)])

y_pred2 = clf.predict_proba(te)

submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred2[:,1]})
submission.to_csv("submission.csv", index=False)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
from sklearn.metrics import precision_recall_curve
from sklearn import cross_validation
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

trainpd=pd.read_csv('../input/train.csv',header=0)
testpd=pd.read_csv('../input/test.csv',header=0)
X_train = trainpd.drop(['ID','TARGET'], axis=1)
clf = ExtraTreesClassifier()
clf = clf.fit(X_train, trainpd['TARGET'])
clf.feature_importances_  

model = SelectFromModel(clf, prefit=True)
train = model.transform(X_train)
train.shape
id_test = testpd['ID']
y_train = trainpd['TARGET'].values
X_test = testpd.drop(['ID'], axis=1).values

rf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=1)

scores = cross_validation.cross_val_score(rf, X_train, y_train, scoring='f1_weighted', cv=5) 
print(scores.mean())
rf.fit(X_train,y_train)
pred = rf.predict_proba(X_test)
submission2 = pd.DataFrame({"ID":testpd.ID, "TARGET":pred[:,1]})
submission2.to_csv("submission.csv", index=False)


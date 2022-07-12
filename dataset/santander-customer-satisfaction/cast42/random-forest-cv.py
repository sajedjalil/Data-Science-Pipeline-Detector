import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score

training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)
print(test.shape)

X = training.iloc[:,:-1]
y = training.TARGET

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

selectK = SelectKBest(f_classif, k=300)
selectK.fit(X, y)
X_sel = selectK.transform(X)

features = X.columns[selectK.get_support()]
print (features)

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_sel,
#   y, random_state=1301, stratify=y, test_size=0.33)
   
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=500, random_state=1301, n_jobs=-1, oob_score=True)

cv = cross_validation.StratifiedKFold(y, n_folds=3,shuffle=True, random_state=1301)

scores = cross_validation.cross_val_score(rfc, X_sel, y, cv=cv, scoring='roc_auc')
print("Auc: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
  
rfc.fit(X_sel, y)      
print('Overall AUC:', roc_auc_score(y, rfc.predict_proba(X_sel)[:,1]))
    
sel_test = selectK.transform(test)    
y_pred = rfc.predict_proba(sel_test)

submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
submission.to_csv("submission_rfc.csv", index=False)



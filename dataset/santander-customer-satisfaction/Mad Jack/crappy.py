import numpy as np 
import pandas as pd
import itertools
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from sklearn import preprocessing


training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)





X = training.iloc[:,:-1]
y = training.TARGET
feat_names_delete = []
for feat_1, feat_2 in itertools.combinations(iterable=X.columns, r=2):
    if np.array_equal(X[feat_1], X[feat_2]):
        feat_names_delete.append(feat_2)
feat_names_delete = np.unique(feat_names_delete)
 
X = X.drop(labels=feat_names_delete, axis=1)
test = test.drop(labels=feat_names_delete, axis=1)
from sklearn.feature_selection import VarianceThreshold

selectVT = VarianceThreshold()
selectVT.fit(X, y)
X_sel = selectVT.transform(X)
sel_test = selectVT.transform(test) 
X_sel[np.isnan(X_sel)] = -999999
sel_test[np.isnan(sel_test)] = -999999


clf = xgb.XGBClassifier(missing=-999999,
                max_depth = 6,
                n_estimators=500,
                learning_rate=0.02, 
                nthread=-1,
                subsample=0.9,
                silent=False,
                colsample_bytree=0.85 
                )
X_fit, X_eval, y_fit, y_eval= train_test_split(X_sel, y, test_size=0.3)
clf.fit(X_sel, y, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])
     
y_pred = clf.predict_proba(sel_test)
submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
submission.to_csv("SANTsubmission.csv", index=False)
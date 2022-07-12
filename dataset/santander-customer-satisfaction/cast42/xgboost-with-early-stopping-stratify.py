import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") #Needed to save figures
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score

training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)
print(test.shape)

X = training.iloc[:,:-1]
y = training.TARGET

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

selectK = SelectKBest(f_classif, k=250)
selectK.fit(X, y)
X_sel = selectK.transform(X)

features = X.columns[selectK.get_support()]
print (features)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_sel,
   y, random_state=1301, stratify=y, test_size=0.33)
clf = xgb.XGBClassifier(max_depth   = 8,
                learning_rate       = 0.02,
                subsample           = 0.9,
                colsample_bytree    = 0.85,
                n_estimators        = 5000)
clf.fit(X_train, y_train, early_stopping_rounds=40, eval_metric="auc",
        eval_set=[(X_test, y_test)])
        
print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X_sel)[:,1]))
    
sel_test = selectK.transform(test)    
y_pred = clf.predict_proba(sel_test)

submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
submission.to_csv("submission.csv", index=False)

mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
ts = pd.Series(clf.booster().get_fscore())
ts.index = ts.reset_index()['index'].map(mapFeat)
ts.order()[-15:].plot(kind="barh", title=("features importance"))

featp = ts.order()[-15:].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)


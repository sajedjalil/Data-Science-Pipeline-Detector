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

# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
training = training.replace(-999999,2)

print('There are {} features.'.format(training.shape[1]))

X = training.iloc[:,:-1]
y = training.TARGET

from sklearn.linear_model import RandomizedLogisticRegression

selector = RandomizedLogisticRegression(n_resampling=400, random_state=1301)
X_sel = selector.fit(X, y)

selected = selector.get_support()
f_selected = [ f for f,s in zip(X.columns, selected) if s]
print('RandomizedLogisticRegression selected {} features:'.format(len(f_selected)))
for f in f_selected:
    print (f)

X_sel = selector.transform(X)
sel_test = selector.transform(test)

X_train, X_test, y_train, y_test = \
   cross_validation.train_test_split(X_sel, y, random_state=1301, stratify=y, test_size=0.35)

clf = xgb.XGBClassifier(max_depth = 5,
                n_estimators=2100,
                learning_rate=0.02, 
                nthread=4,
                subsample=0.95,
                colsample_bytree=0.85, 
                seed=4242)
clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="auc",
        eval_set=[(X_test, y_test)])
        
print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X_sel, ntree_limit=clf.best_iteration)[:,1]))

y_pred = clf.predict_proba(sel_test, ntree_limit=clf.best_iteration)

submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
submission.to_csv("submission.csv", index=False)



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from collections import defaultdict

training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)
print(test.shape)

X = training.iloc[:,:-1]
y = training.TARGET

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

selectK = SelectKBest(f_classif, k=300)
selectK.fit(X, y)
X_sel = selectK.transform(X)

features = X.columns[selectK.get_support()]
print (features)

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_sel,
#   y, random_state=1301, stratify=y, test_size=0.33)
   
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, random_state=1301, n_jobs=-1,
   criterion='gini', class_weight='balanced')

scores = defaultdict(list)

y = np.array(y.astype(int)).ravel()

# Based on http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
#crossvalidate the scores on a number of different random splits of the data
for train_idx, test_idx in cross_validation.StratifiedShuffleSplit(y, 3, .3):
    X_train, X_test = X_sel[train_idx], X_sel[test_idx]
    Y_train, Y_test = y[train_idx], y[test_idx]
    r = rfc.fit(X_train, Y_train)
    auc = roc_auc_score(Y_test, rfc.predict(X_test))
    for i in range(X_sel.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_auc = roc_auc_score(Y_test, rfc.predict(X_t))
        scores[features[i]].append((auc-shuff_auc)/auc)
print ("Features sorted by their score:")
print (sorted([(round(np.mean(score), 4), feat) for
              feat, score in scores.items()], reverse=True))
              
features = [feat for feat, score in scores.items() if score > 0.0]

print('Selected features by crossvalidation with RFC with 100 trees:')
print(features)
X_sel_s = X_sel[features]

rfc = RandomForestClassifier(n_estimators=1000, random_state=1301, n_jobs=-1,
   criterion='gini', class_weight='balanced')
rfc.fit(X_sel_s, y)

sel_test = test[features]    
y_pred = rfc.predict_proba(sel_test)

submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
submission.to_csv("submission_rfc.csv", index=False)



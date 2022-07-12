# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score

x_train = pd.read_csv("../input/train.csv")
x_test = pd.read_csv("../input/test.csv").values

target = x_train.iloc[:, -1].values
X = x_train.iloc[:, 1:-1].values

pca = PCA(n_components=0.8)
X_new = pca.fit_transform(X)

cv_step = np.arange(23, 28)
clfs = []; scores = [];
for i in cv_step:
    clf = SVC(class_weight={1: i})
    clfs.append(clf)
    kf = KFold(len(X_new), n_folds=5)
    final_score = 0
    for train_fold, test_fold in kf:
        clf.fit(X_new[train_fold], target[train_fold])
        score = roc_auc_score(target[test_fold], clf.predict_proba(x_train[test_fold]))
        final_score += score
    scores.append(final_score / 5.)
    print(scores)
best_clf = clfs[np.argmax(scores)]
best_clf.fit(X_new, target)
result = best_clf.predict_proba(pca.transform(x_test))

result = pd.DataFrame({"ID": x_test[:, 0].astype('int'), "TARGET": result})
result.to_csv('submission.csv', index=False)
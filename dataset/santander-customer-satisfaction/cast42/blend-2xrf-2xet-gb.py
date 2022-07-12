# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from __future__ import division
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Based on Blending code from Emanuele Olivetti

"""
Blending {RandomForests, ExtraTrees, GradientBoosting} + stretching to
[0,1]. The blending scheme is related to the idea Jose H. Solorzano
presented here:
http://www.kaggle.com/c/bioresponse/forums/t/1889/question-about-the-process-of-ensemble-learning/10950#post10950
'''You can try this: In one of the 5 folds, train the models, then use
the results of the models as 'variables' in logistic regression over
the validation data of that fold'''. Or at least this is the
implementation of my understanding of that idea :-)

The predictions are saved in test.csv.

Copyright 2012, Emanuele Olivetti.
BSD license, 3 clauses.
"""

import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':

    np.random.seed(1301) # seed to shuffle the train set

    n_folds = 5
    verbose = True
    shuffle = False

    training = pd.read_csv("../input/train.csv", index_col=0)
    test = pd.read_csv("../input/test.csv", index_col=0)
    print(training.shape)
    print(test.shape)

    # Replace -999999 in var3 column with most common value 2 
    # See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
    # for details
    training = training.replace(-999999,2)

    X = training.iloc[:,:-1]
    y = training.TARGET

    # remove constant columns
    remove = []
    for col in X.columns:
        if X[col].std() == 0:
            remove.append(col)

    X.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif

    selectK = SelectKBest(f_classif, k=220)
    selectK.fit(X, y)
    X_sel = selectK.transform(X)

    features = X.columns[selectK.get_support()]
    print (features)
    sel_test = selectK.transform(test) 
    X, y, X_submission = np.array(X_sel), np.array(y.astype(int)).ravel(), np.array(sel_test)

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini', class_weight='balanced'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy', class_weight='balanced'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

    print ("Creating train and test sets for blending.")
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    skf = cross_validation.StratifiedKFold(y, n_folds)
    
    for j, clf in enumerate(clfs):
        print (j, clf)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], n_folds))
        for i, (train, testidx) in enumerate(skf):
            print ("Fold", i)
            X_train, y_train = X[train], y[train]
            X_test, y_test = X[testidx], y[testidx]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[testidx, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(axis=1)

    print ("Blending.")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]

    print ("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print ("Saving Results.")
    submission = pd.DataFrame({"ID":test.index, "TARGET":y_submission})
    submission.to_csv("submission_2xRF2xETGB001.csv", index=False)
    
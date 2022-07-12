# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def load_csv():
    with open("../input/train.csv") as f:
        f.readline()
        data = np.loadtxt(f, delimiter=',')
    with open("../input/test.csv") as f:
        f.readline()
        Xtest = np.loadtxt(f, delimiter=',')
    y = data[:, -1]
    X = data[:, :-1]
    return X, y, Xtest


def remove_constants(X, Xtest):
    # Remove IDs
    Xnew = deepcopy(X)
    Xtestnew = deepcopy(Xtest)
    ids = Xnew[:, 0]
    idstest = Xtestnew[:, 0]
    Xnew = Xnew[:, 1:]
    Xtestnew = Xtestnew[:, 1:]
    # Remove constants
    num_features = Xnew.shape[1]
    Xvar = np.var(Xnew, 0)
    constant_idx = (Xvar == np.zeros(num_features))
    Xnew = Xnew[:, ~constant_idx]
    Xtestnew = Xtestnew[:, ~constant_idx]
    return Xnew, Xtestnew, ids, idstest


def add_summary_features(X, Xtest):
    # Add column with sum of zero counts
    Xnew = deepcopy(X)
    Xtestnew = deepcopy(Xtest)
    zerocol = np.zeros((Xnew.shape[0], 1))
    zerocoltest = np.zeros((Xtestnew.shape[0], 1))
    for idx in range(Xnew.shape[0]):
        zerocol[idx, 0] = np.sum(X[idx, :] == 0)
    for idx in range(Xtestnew.shape[0]):
        zerocoltest[idx, 0] = np.sum(Xtest[idx, :] == 0)
    Xnew = np.hstack((Xnew, zerocol))
    Xtestnew = np.hstack((Xtestnew, zerocoltest))
    # Add first two PCA features
    Xnorm = normalize(X)
    Xtestnorm = normalize(Xtest)
    pca = PCA(n_components=2)
    Xpca = pca.fit_transform(Xnorm)
    Xtestpca = pca.fit_transform(Xtestnorm)
    Xnew = np.hstack((Xnew, Xpca[:, :2]))
    Xtestnew = np.hstack((Xtestnew, Xtestpca[:, :2]))
    return Xnew, Xtestnew


X, y, Xtest = load_csv()

skf = cross_validation.StratifiedKFold(y, n_folds=10, shuffle=True)
aucs = []
for train_index, test_index in skf:
    Xtrain, Xvalid = X[train_index, :], X[test_index, :]
    ytrain, yvalid = y[train_index], y[test_index]
    num_features = X.shape[1]
    max_features = int(np.ceil(np.sqrt(num_features)))
    fclf = RandomForestClassifier(max_depth=None, n_estimators=100,
                                  max_features=max_features, verbose=0,
                                  n_jobs=-1)
    fclf.fit(Xtrain, ytrain)
    probs = fclf.predict_proba(Xvalid)
    auc_score = roc_auc_score(yvalid, probs[:, -1])
    aucs.append(auc_score)
print(np.mean(aucs))

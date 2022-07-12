# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def logloss(attempt, actual, epsilon=1.0e-15):
    """Logloss, i.e. the score of the bioresponse competition.
    """
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)
    return - np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt))



np.random.seed(0) # seed to shuffle the train set

n_folds = 2
verbose = True
shuffle = False

# X, y, X_submission = load_data.load()
train_df = pd.read_csv("../input/train.csv", index_col=None)
test_df = pd.read_csv("../input/test.csv", index_col=None)

train_df = train_df.replace(-999999,2)
X = train_df.iloc[:, 1:-1].values
y = train_df['TARGET'].values

X_submission= test_df.iloc[:, 1:].values

if shuffle:
    idx = np.random.permutation(y.size)
    X = X[idx]
    y = y[idx]

skf = list(StratifiedKFold(y, n_folds))

clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.8, max_depth=6, n_estimators=50, verbose = 1)]

print("Creating train and test sets for blending.")

dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    print(j, clf)
    dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        print("Fold", i)
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:,1]
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

print("Blending.")
# clf = GradientBoostingClassifier(learning_rate = 0.03, n_estimators = 250, max_depth = 7, min_samples_leaf =7, verbose = 1, loss = "deviance")
# clf = SVC(kernel = "linear", probability = True)
clf = LogisticRegression(solver = "sag")
clf.fit(dataset_blend_train, y)
y_submission = clf.predict_proba(dataset_blend_test)[:,1]


print("Saving Results.")
submission = pd.DataFrame({"ID":test_df['ID'].values, "TARGET":y_submission})
submission.to_csv("submission.csv", index=False)
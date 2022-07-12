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

from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.ensemble._gradient_boosting import predict_stage
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv("../input/train.csv", index_col=None)
test_df = pd.read_csv("../input/test.csv", index_col=None)

train_df = train_df.replace(-999999,2)
X = train_df.iloc[:, 1:-1].values
y = train_df['TARGET'].values

X_test = test_df.iloc[:, 1:].values
pred_list = []
"""
for depth in [4, 5, 6]:
    clf = GradientBoostingClassifier(learning_rate = 0.03, n_estimators = 250, max_depth = depth, min_samples_leaf =7,max_features = 0.7, verbose = 1, loss = "deviance")
    clf.fit(X, y)
    pred_list.append(clf.predict_proba(X_test)[:,1])
"""
print("A")
clf = LogisticRegression(solver = "sag")
print("B")
clf.fit(X, y)
print("C")
pred_list.append(clf.predict_proba(X_test)[:,1])

pred = np.mean(np.array(pred_list), axis = 0)

submission = pd.DataFrame({"ID":test_df['ID'].values, "TARGET":pred})
submission.to_csv("submission.csv", index=False)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
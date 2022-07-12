# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import seaborn as sns
import pandas as pd
import sklearn 
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_selection import VarianceThreshold

# Any results you write to the current directory are saved as output.
import xgboost as xgb
print(xgb.__version__)
SEED = 1337

data = pd.read_csv("../input/train.csv", index_col="ID")
X, y = data.drop("TARGET", axis=1), data['TARGET']
drop_cols = [c for c in X.columns if X[c].std() == 0]
print(X.shape)
X.drop(drop_cols, axis=1, inplace=True)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(normalize(X), y, test_size=0.15, stratify=y, random_state=SEED)

m = xgb.XGBClassifier(max_depth=6, n_estimators=300, learning_rate=0.05)
m.fit(X_train, y_train)
_proba = m.predict_proba(X_test)
print(roc_auc_score(y_test, _proba[:,1]))


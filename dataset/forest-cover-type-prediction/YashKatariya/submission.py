from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# load data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# remove constant columns
remove = []
for col in df_train.columns:
    if df_train[col].std() == 0:
        remove.append(col)

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
c = df_train.columns
for i in range(len(c)-1):
    v = df_train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,df_train[c[j]].values):
            remove.append(c[j])

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

y_train = df_train['Cover_Type'].values
y_train = [x-1 for x in y_train]
X_train = df_train.drop(['Id','Cover_Type'], axis=1).values


id_test = df_test['Id']
X_test = df_test.drop(['Id'], axis=1).values

print(len(X_train), len(y_train))

# length of dataset
len_train = len(X_train)
len_test  = len(X_test)

# classifier
clf = xgb.XGBClassifier(missing=np.nan, max_depth=7, n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)

X_fit, X_eval, y_fit, y_eval= train_test_split(X_train, y_train, test_size=0.3)

# fitting
clf.fit(X_train, y_train, early_stopping_rounds=40, eval_metric="merror", eval_set=[(X_eval, y_eval)])

# predicting
y_pred= clf.predict(X_test)
y_pred = [x+1 for x in y_pred]

submission = pd.DataFrame({"Id":id_test, "Cover_Type":y_pred})
submission.to_csv("submission.csv", index=False)


print('Completed!')
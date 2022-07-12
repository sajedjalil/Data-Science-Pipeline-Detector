# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectFpr
from sklearn.metrics import roc_auc_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

##################
# IMPORTING DATA #
##################

train_raw = pd.read_csv('../input/train.csv')
test_raw = pd.read_csv('../input/test.csv')

#################
# DATA CLEANING #
#################

columnstrain = train_raw.columns
columnstest = test_raw.columns
len(columnstrain) # 371 columns
len(columnstest) # 370 columns

# Removing constant columns & duplicate columns
removeColumns1 = []
for col in train_raw.columns:
    if train_raw[col].std() == 0:
        removeColumns1.append(col)

len(removeColumns1) #34 variables are constants

train_raw.drop(removeColumns1, axis=1, inplace=True)
test_raw.drop(removeColumns1, axis=1, inplace=True)

removeColumns2 = []
cols = train_raw.columns
for i in range(len(cols)-1):
    v = train_raw[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,train_raw[cols[j]].values):
            removeColumns2.append(cols[j])
            
train_raw.drop(removeColumns2, axis=1, inplace=True)
test_raw.drop(removeColumns2, axis=1, inplace=True)

train_target = train_raw.TARGET.values
test_ids = test_raw.ID

train_derived = train_raw.drop(['ID','TARGET'], axis=1)
test_derived = test_raw.drop(['ID'], axis=1)

############
# ANALYSIS #
############

pval = SelectFpr(alpha = 0.001) 
train_variables = pval.fit_transform(train_derived, train_target)

keepColumns = pval.get_support(indices = True)

cols = train_derived.columns
removeColumns = []
for i in range(len(cols)):
    if i not in keepColumns:
        removeColumns.append(cols[i])
test_variables = test_derived.drop(removeColumns, axis=1).values


clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=550, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)
X_train, X_test, y_train, y_test = train_test_split(train_variables, train_target, test_size=0.3)

# fitting
clf.fit(train_variables, train_target, eval_metric="auc", early_stopping_rounds=20, eval_set=[(X_test, y_test)])

print('AUC:', roc_auc_score(train_target, clf.predict_proba(train_variables)[:,1]))

test_predictions = clf.predict_proba(test_variables)[:,1]

submission = pd.DataFrame({"ID":test_ids, "TARGET":test_predictions})
submission.to_csv("submission.csv", index=False)
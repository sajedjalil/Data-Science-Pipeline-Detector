# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

print('Load data...')
train = pd.read_csv("../input/train.csv")
train_id = train['ID'].values
target = train['TARGET'].values
train = train.drop(['ID','TARGET'],axis=1)

test = pd.read_csv("../input/test.csv")
test_id = test['ID'].values
test = test.drop(['ID'],axis=1)

#removing outliers
train = train.replace(-999999,2)
test = test.replace(-999999,2)

# adding zero counts

train["zeroes"] = (train == 0).astype(int).sum(axis=1)
test["zeroes"] = (test == 0).astype(int).sum(axis=1)

# remove constant columns (std = 0)
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

print(train.shape, test.shape)

# remove duplicated columns
remove = []
cols = train.columns
for i in range(len(cols)-1):
    v = train[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,train[cols[j]].values):
            remove.append(cols[j])

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)


# Feature selection 
#ROUND ONE
Cols = train.columns.values.tolist()
clf = GradientBoostingClassifier(random_state=1729)
selector = clf.fit(train, target)
importances = selector.feature_importances_
fs = SelectFromModel(selector, prefit=True)
train = fs.transform(train)
test = fs.transform(test)
print(train.shape, test.shape)

selectedCols = train.shape[1]
sortedCols = [col for importance, col  in sorted(zip(importances, Cols))]
sortedCols = sortedCols[0:selectedCols]
train = pd.DataFrame(train)
test = pd.DataFrame(test)
train.columns = sortedCols
test.columns = sortedCols

#Selecting Percentile Changes from feature to feature

for i in range(len(sortedCols)):
	for j in range(len(sortedCols)):
		colName = sortedCols[i]+"_SUBTRACT_"+sortedCols[j]+"DIVIDE"+sortedCols[i]
		train[colName] = (train[sortedCols[i]]-train[sortedCols[j]])/train[sortedCols[i]]
		test[colName] = (test[sortedCols[i]]-test[sortedCols[j]])/test[sortedCols[i]]

train = train.replace(np.inf, 999999)
train = train.replace(-np.inf, -999999)
train = train.replace(np.nan, -1)
test = test.replace(np.inf, 999999)
test = test.replace(-np.inf, -999999)
test = test.replace(np.nan, -1)

#ROUND TWO
Cols = train.columns.values.tolist()
clf = GradientBoostingClassifier(random_state=1729)
selector = clf.fit(train, target)
importances = selector.feature_importances_
fs = SelectFromModel(selector, prefit=True)
train = fs.transform(train)
test = fs.transform(test)
print(train.shape, test.shape)

selectedCols = train.shape[1]
sortedCols = [col for importance, col  in sorted(zip(importances, Cols))]
sortedCols = sortedCols[0:selectedCols]
print(sortedCols)

predictedResult = np.zeros(train.shape[0])
kf = KFold(train.shape[0], n_folds=10)
testPred = []
for trainIndex, testIndex in kf:
    trainFold, testFold = train[trainIndex], train[testIndex]
    trainFoldTarget, testFoldTarget = target[trainIndex], target[testIndex]
    xgbc = xgb.XGBClassifier(n_estimators = 560,learning_rate = 0.0202047,max_depth = 5,subsample = 0.6815,colsample_bytree = 0.701)
    xgbc.fit(trainFold,trainFoldTarget)
    xgbpred =xgbc.predict_proba(testFold)[:,1]
    testPred.append(xgbc.predict_proba(test)[:,1])
    predictedResult[testIndex] = xgbpred
    print(roc_auc_score(testFoldTarget, xgbpred))

print(roc_auc_score(target, predictedResult))
testPred = np.average(np.array(testPred), axis =0)
pd.DataFrame({"ID": test_id, "TARGET": testPred}).to_csv('submission.csv',index=False)
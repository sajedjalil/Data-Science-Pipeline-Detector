# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.cross_validation import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.
np.random.seed(1234)


trainDF = pd.read_csv("../input/train.csv")
testDF = pd.read_csv('../input/test.csv')
print (trainDF.shape)
print (testDF.shape)
# #Grabbing stuff
# targets = trainDF.TARGET
# trainDF = trainDF.drop(['ID', 'TARGET'], axis=1)
# test_ids = testDF.ID
# testDF = testDF.drop(['ID'], axis=1)

# var15 = testDF['var15']
# saldo_medio_var5_hace2 = testDF['saldo_medio_var5_hace2']
# saldo_var33 = testDF['saldo_var33']
# var38 = testDF['var38']
# V21 = testDF['var21']

# #Replace -999999 in var3 column with most common value of 2
# trainDF = trainDF.replace(-999999,2)
# testDF = testDF.replace(-999999,2)

# #Summing 0 per row
# trainDF['n0'] = (trainDF == 0).sum(axis=1)
# testDF['n0'] = (testDF == 0).sum(axis=1)

# # remove constant columns
# colsToRemove = []
# for col in trainDF.columns:
#     if trainDF[col].std() == 0:
#         print("Deleting ", col)
#         colsToRemove.append(col)

# # remove duplicate columns
# columns = trainDF.columns
# for i in range(len(columns)-1):
#     v = trainDF[columns[i]].values
#     for j in range(i+1,len(columns)):
#         if np.array_equal(v,trainDF[columns[j]].values):
#             print("Deleting ", columns[j])
#             colsToRemove.append(columns[j])

# trainDF.drop(colsToRemove, axis=1, inplace=True)
# testDF.drop(colsToRemove, axis=1, inplace=True)

# print(testDF.shape)
# print(trainDF.shape)

# # clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=1234)
# clf = xgb.XGBClassifier(max_depth = 5,n_estimators=560,learning_rate=0.0202048, nthread=4,subsample=0.6815,colsample_bytree=0.701, seed=1234)
# # fitting
# clf.fit(trainDF, targets, eval_metric="auc", eval_set=[(trainDF, targets)])

# test_pred = clf.predict_proba(testDF)[:,1]
# test_pred[var15 < 23] = 0
# test_pred[saldo_medio_var5_hace2 > 160000]=0
# test_pred[saldo_var33 > 0]=0
# test_pred[var38 > 3988596]=0
# test_pred[V21>7500]=0

# submission = pd.DataFrame({"ID":test_ids, "TARGET":test_pred})
# submission.to_csv("submission2.csv", index=False)
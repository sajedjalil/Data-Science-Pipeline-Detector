#Code for removing constant and identical features in Python
import os
import numpy as np
from sklearn.feature_selection import RFE, VarianceThreshold
import itertools
import pandas as pd

train_df = pd.read_csv('../input/train.csv', header=0)
test_df = pd.read_csv('../input/test.csv', header=0)

xTrain = train_df.drop(['ID','TARGET'], axis = 1)
xTrain = np.array(xTrain)

xTest = test_df.drop('ID',axis = 1)
xTest = np.array(xTest)

selector = VarianceThreshold(0)
xTrain = selector.fit_transform(xTrain)
xTest = selector.transform(xTest)

col_identical = set()
for pair in itertools.combinations(range(np.shape(xTrain)[1]),2):
    if np.array_equal(xTrain[:,pair[0]],xTrain[:,pair[1]]):
        col_identical.add(pair[0])

for i in col_identical:
    xTrain = np.delete(xTrain,i,1)
    xTest = np.delete(xTest,i,1)

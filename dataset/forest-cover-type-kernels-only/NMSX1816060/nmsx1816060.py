# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

traindata = pd.read_csv('../input/train.csv')
testdata = pd.read_csv('../input/test.csv')
features = traindata.copy().iloc[:, 1:-1]
labels = traindata.iloc[:, -1]
rf = ExtraTreesClassifier(n_estimators=350)
rf.fit(features,labels)
df = pd.DataFrame({"Id":testdata['Id'],"Cover_Type": rf.predict(testdata.iloc[:, 1:])})
df.to_csv("output.csv",index=False)
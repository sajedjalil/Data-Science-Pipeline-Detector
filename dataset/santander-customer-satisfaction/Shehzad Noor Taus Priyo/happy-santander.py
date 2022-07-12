# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import csv
from sklearn.tree import DecisionTreeClassifier

trainFile = open("../input/train.csv")
trainInfo = csv.DictReader(trainFile)

features_train = []
labels_train = []
rowNo = 0

for row in trainInfo:
    features_train.append([])
    for key in row.keys():
        if key != "TARGET":
            features_train[rowNo].append(float(row[key]))
        else:
            labels_train.append(int(row[key]))
    rowNo += 1


clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

testFile = open("../input/test.csv")
testInfo = csv.DictReader(testFile)

features_test = []
labels_test = []
rowNo = 0

for row in testInfo:
    features_test.append([])
    for key in row.keys():
        features_test[rowNo].append(float(row[key]))
    rowNo += 1

pred = clf.predict(features_test)

for row in testInfo:
    print(row['ID'])




from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets, model_selection, naive_bayes
import pandas as pd
import numpy as np
import csv

filename = r'../input/train.csv'
dataset = pd.read_csv(filename)
dataset = dataset.values
x = dataset[:,:-1]
y = []
for i in range(len(dataset)):
    y.append(dataset[i,-1])
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01)

cls = naive_bayes.MultinomialNB()
cls.fit(x, y)

# clf = RandomForestClassifier(n_estimators= 50,max_depth=50)
# clf.fit(x, y)
filename = r"../input/test.csv"
x_test = pd.read_csv(filename)
x_test = x_test.values
y_predict = clf.predict(x_test)
csvfile = open("sample_submission1.csv", "w", encoding="gb18030", newline="")
writer = csv.writer(csvfile)
writer.writerow(["ID", "Cover_Type"])
for i in range(len(y_predict)):
    x = [x_test[i,0], y_predict[i]]
    writer.writerow(x)
csvfile.close()
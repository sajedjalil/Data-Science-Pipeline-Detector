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
dataset = pd.read_csv('../input/train.csv')
X = dataset.iloc[:, 1:-1].values
print (X)
y = dataset.iloc[:, 55].values
print (y)

dataset_test = pd.read_csv('../input/test.csv')
print (dataset_test)
X_test = dataset_test.iloc[:, 1:].values
print (X_test)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)
print (X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X, y)
print (classifier)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print (y_pred)
y_ind = dataset_test.iloc[:, 0].values.reshape(565892, 1)
print (y_ind)
y_predHor = y_pred.reshape(565892, 1)
print (y_predHor)
output = np.concatenate((y_ind, y_predHor), axis=1)
print (output)
np.savetxt("output.csv", output, delimiter=",")













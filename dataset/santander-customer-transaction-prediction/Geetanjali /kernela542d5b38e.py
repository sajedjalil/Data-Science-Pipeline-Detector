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

df = pd.read_csv('../input/train.csv')

trainData = pd.read_csv('../input/train.csv',index_col='ID_code')
trainData.head()

X = trainData.drop("target", axis=1)
Y = trainData["target"]

from sklearn.model_selection import train_test_split
#building logistic regression model
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20, random_state=101)
                                                    
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))

# Read the test data
test = pd.read_csv('../input/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test.drop("target", axis=1)
# Use the model to make predictions
predicted_transactions = logmodel.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_transactions)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))


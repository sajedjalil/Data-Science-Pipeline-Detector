# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

x_train = train["question_text"]
y_train = train["target"]
x_test = test["question_text"]
cv= CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

model= LogisticRegression(penalty = "l1", C = 1.25, class_weight = "balanced")

#model= SVC(kernel = 'linear', random_state = 0)

model.fit(x_train, y_train)
pred = model.predict(x_test)

print("Train Accuracy score is:", model.score(x_train, y_train))

submission = pd.DataFrame({'qid':test['qid'],'prediction':pred})
submission.to_csv('submission.csv', index=False)










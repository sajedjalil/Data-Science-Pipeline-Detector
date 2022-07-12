# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import metrics



train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

Id=test['Id']
y=train['Cover_Type']
train=train.drop(['Id','Cover_Type'],1)
test=test.drop(['Id'],1)
x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.1, random_state=1)

RF = ensemble.RandomForestClassifier(n_estimators=200,class_weight='balanced',n_jobs=2,random_state=1)
RF.fit(x_train,y_train)
pred=RF.predict(x_test)
acc=RF.score(x_test,y_test)
print(acc)
ct=RF.predict(test)
output=pd.DataFrame(Id)
output['Cover_Type']=ct
output.head()
output.to_csv("submission.csv",index=False)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
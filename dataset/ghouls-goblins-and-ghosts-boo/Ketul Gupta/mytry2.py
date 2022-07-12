# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

train= pd.read_csv('../input/train.csv')
# print (train.head())
# print (train['type'])
print (train['color'].unique())
train['color'].replace(['clear','green','black','white','blue','blood'] , [0,1,2,3,4,5] , inplace=True)
print (train['color'].unique())
        
train2=train
train2= train2.drop(['type', 'id'],axis=1)
print (train2.shape)

test= pd.read_csv('../input/test.csv')
test['color'].replace(['clear','green','black','white','blue','blood'] , [0,1,2,3,4,5] , inplace=True)
test1=test.drop('id',axis=1)
print (test.shape)
# gnb = GaussianNB()   #0.7
# gnb = tree.DecisionTreeClassifier()  #0.6
gnb=RandomForestClassifier(n_estimators=40)
y_pred = gnb.fit(train2,train['type']).predict(test1)

submission = pd.read_csv("../input/sample_submission.csv")
#submission.iloc[:, 1] = np.exp(bst.predict(test_xgb))
submission.iloc[:, 1] = y_pred
submission.to_csv('sub3.csv', index=None)

# Number of mislabeled points out of a total 150 points : 6
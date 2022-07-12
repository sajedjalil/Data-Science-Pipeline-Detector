# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing  
from sklearn import svm 

# Input data files are available in the "../input/" directory.
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

#Prepare data
#Target data
y = train_data.species
#Train data
X_train = train_data.drop('species',1)
X_train = X_train.drop('id',1)
X_test = test_data.drop('id',1)

#Convert data to [-1,1] space
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)

#Init algo
clf = svm.SVC(kernel='linear',probability=True)

#Train data
clf.fit(X_train_scale,y)

#Predict
pred = clf.predict_proba(X_test_scale)

# Any results you write to the current directory are saved as output.
df1 = pd.DataFrame(y)
df2 = pd.DataFrame(pred)
df3 = test_data.id

df = df3.append(df2)

print(df)

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
train1 = pd.read_csv("../input/train.csv")
output1 = pd.read_csv("../input/trainLabels.csv")
test = pd.read_csv("../input/test.csv")
X_train = train1.values
y_train = output1.values
X_test = test.values
 
from sklearn.preprocessing import Imputer
imputer = Imputer()
imputer = imputer.fit(X_train)
X_train = imputer.transform(X_train)



from sklearn.ensemble import RandomForestRegressor
classifier = RandomForestRegressor(n_estimators = 300,random_state = 0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
classifier.score(X_train,y_train)
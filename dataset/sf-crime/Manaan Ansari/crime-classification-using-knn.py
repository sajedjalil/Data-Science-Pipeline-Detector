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

#import libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#make loss function to test 

def llfun(act, pred):
    return (-(~(act == pred)).astype(int) * math.log(1e-15)).sum() / len(act)


train = pd.read_csv('../input/train.csv', parse_dates=['Dates'])[['X', 'Y', 'Category']]

X = train.iloc[:,0:-1].values
y = train.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Separate test and train set out of orignal train set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fit
logloss = []
for i in range(1, 50, 1):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    
    # Predict on test set
    y_pred = knn.predict(X_test)
    
    # Logloss
    logloss.append(llfun(y_test, y_pred))

plt.plot(logloss)
plt.savefig('loss.png')

# Submit for K=30
test = pd.read_csv('../input/test.csv', parse_dates=['Dates'])
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X, y)
x_test = test[['X', 'Y']]
outcomes = labelencoder_y.inverse_transform(knn.predict(x_test))

submit = pd.DataFrame({'Id': test.Id.tolist()})
for category in labelencoder_y.classes_:
    submit[category] = np.where(outcomes == category, 1, 0)


submit.to_csv('KNN.csv', index = False)
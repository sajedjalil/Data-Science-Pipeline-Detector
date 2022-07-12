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


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
target = pd.read_csv('../input/sample_submission.csv')




#
X = train_data.iloc[:,2:202].values
y = train_data.iloc[:,1].values


# Splitting the dataset into the Training set and Test set



##Creating an Object of StandardSacaler
#Standard Skale for Train data
obj_standart = StandardScaler()
X = obj_standart.fit_transform(X)


# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X = lda.fit_transform(X, y)



from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(X)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

#################################################################################

X_t = test_data.iloc[:,1:202]


t_target=target.iloc[:,1]



X_t = obj_standart.fit_transform(X_t)



X_t = lda.transform(X_t)
y_t = classifier.predict(X_t)


target['target'] = y_t
#target.drop(['target'],axis=1, inplace = True)

target.to_csv('submission1.csv', index=False)

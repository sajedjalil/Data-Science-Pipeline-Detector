# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split
# FULL NOTEBOOK CAN BE FOUND on the following URL: https://github.com/gsam1/dsml-projects/blob/master/5_Playground/Kannada_MNIST.ipynb
# ## 1. Loading and Exploring the data.

train_df = pd.read_csv('../input/Kannada-MNIST/train.csv')
test_df = pd.read_csv('../input/Kannada-MNIST/test.csv')


# ## 2. The Train-Test Split

# Train-Test split has been chosen (arbitrarily) to be 20%.


X, y = train_df.iloc[:, 1:], train_df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## 3. Models

# ### 3.1 The Shallows

# utils
from sklearn.model_selection import KFold, cross_val_score
# RF
from sklearn.ensemble import RandomForestClassifier
# SVM
from sklearn.svm import SVC
# KNN
from sklearn.neighbors import KNeighborsClassifier
# Optimization
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# #### 3.1.1 RF
# Training an Random Forest to set a baseline for the CNNs used later.



rf = RandomForestClassifier(n_estimators = 500, max_features='sqrt', min_samples_leaf = 5, random_state = 42, n_jobs=-1)
rf_scores = cross_val_score(rf, X_train, y_train, cv=5)

print('Training...')
rf.fit(X_train, y_train)
print('Generating Predictions...')
X_subm = test_df.iloc[:, 1:].values
y_subm = rf.predict(X_subm)

indices = [i for i in range(0, y_subm.shape[0])]
predictions_df = pd.DataFrame({'id':indices, 'label': y_subm})

predictions_df.to_csv('submissions.csv', index=False)



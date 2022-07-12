# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

import os
#print(os.listdir("../input"))

import time
start_time = time.clock()

dataset = pd.read_csv('../input/application_train.csv')
dataset_test = pd.read_csv('../input/application_test.csv')

X_train = dataset.iloc[:, dataset.columns != 'TARGET'] #.values
y_train = dataset.iloc[:, dataset.columns == 'TARGET'] #.values
X_test = dataset_test.iloc[:,dataset_test.columns != 'TARGET'] #.values

# impute missing values for numeric variables
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train.select_dtypes(exclude=["bool_","object_"]))
X_train_numeric = imputer.transform(X_train.select_dtypes(exclude=["bool_","object_"]))
X_train_numeric = pd.DataFrame(X_train_numeric,columns = X_train.select_dtypes(exclude=["bool_","object_"]).columns)

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test.select_dtypes(exclude=["bool_","object_"]))
X_test_numeric = imputer.transform(X_test.select_dtypes(exclude=["bool_","object_"]))
X_test_numeric = pd.DataFrame(X_test_numeric,columns = X_test.select_dtypes(exclude=["bool_","object_"]).columns)

# create dummy variables for the categorical variables
X_train_categorical = pd.get_dummies(X_train.select_dtypes(exclude=["number"]), drop_first=True)
X_test_categorical = pd.get_dummies(X_test.select_dtypes(exclude=["number"]), drop_first=True)

# merge numeric and categorical predictors
X_train = X_train_categorical.merge(X_train_numeric,left_index=True,right_index=True)
X_test = X_test_categorical.merge(X_test_numeric,left_index=True,right_index=True)

# Get missing columns in the training test
missing_cols = set(X_train.columns ) - set(X_test.columns)
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    X_test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
X_test = X_test[X_train.columns]

# Feature Scaling s
sc_X = StandardScaler()
X_train = pd.DataFrame(sc_X.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(sc_X.fit_transform(X_test), columns = X_test.columns)

# important features?

# Fitting Random Forest Classification to the Training set

classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy') #MAX estimators??
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

pd.DataFrame({'SK_ID_CURR':dataset_test.SK_ID_CURR,'TARGET':y_pred}).describe()
pd.DataFrame({'SK_ID_CURR':dataset_test.SK_ID_CURR,'TARGET':y_pred}).hist()

#submit file for test data
predictions = pd.DataFrame({'SK_ID_CURR':dataset_test.SK_ID_CURR,'TARGET':y_pred})

# prepare the csv file
predictions.to_csv('prediction.csv',index=False)

run_time = time.clock()-start_time
print('Total run time = ' + str(run_time))













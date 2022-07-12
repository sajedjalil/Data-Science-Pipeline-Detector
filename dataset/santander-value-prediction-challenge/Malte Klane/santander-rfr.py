# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


import time
start_time = time.clock()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# reading the dataset
dataset = pd.read_csv('../input/train.csv')
dataset_test = pd.read_csv('../input/test.csv')

X = dataset.iloc[:,2:4993]#.values
y = dataset.iloc[:, 1]#.values
X_test = dataset_test.iloc[:,1:4992]#.values

# Feature Scaling 
X_columns = X.columns
X_columns_test = X_test.columns
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X = pd.DataFrame(X,columns= X_columns)
X_test = sc_X.transform(X_test)
X_test = pd.DataFrame(X_test,columns= X_columns_test)

# build the model (Random Forest Regression)
regressor = RandomForestRegressor(n_estimators = 10) 
regressor.fit(X, y)

# get feature importances
importances = list(regressor.feature_importances_)
len(importances) #4991
len(X) #4459

# add row with importances
importances_df = pd.DataFrame(importances)
importances_transposed = importances_df.T
importances_df_transposed = pd.DataFrame(importances_transposed)
importances_df_transposed.columns = X.columns
    #X_importances = X.append(importances_df_transposed)

# create list of features with importance of 0.0
list_unimportant_features = importances_df_transposed.loc[:, (importances_df_transposed == 0).any(axis=0)].columns

# move features into list until sum of importances is 0.95
importances_df_T = importances_df_transposed.loc[:, (importances_df_transposed != 0).any(axis=0)].T
importances_df_T.columns = ['importance']
importances_df_T.sort_values(by='importance', ascending=False)

for x in range(len(importances_df_T)):     
  if (importances_df_T.iloc[0:x,0].sum()) >= 0.95:
    number_of_important_features = x #print(x)
    break

list_unimportant_features.join(importances_df_T.iloc[number_of_important_features:(len(importances_df_T)),0].index.values)

# drop features of test and train data according to importance list

for x in range(len(list_unimportant_features)):
    X.drop(list_unimportant_features.values[x], axis = 1, inplace = True)
    X_test.drop(list_unimportant_features.values[x], axis = 1, inplace = True)

# increase the number of trees in regressor and rebuilt model with pruned data
regressor_pruned = RandomForestRegressor(n_estimators = 200)   # change to MAX
regressor_pruned.fit(X, y)

# make predictions
y_pred = regressor_pruned.predict(X_test)

#submit file for test data
predictions = pd.DataFrame({'ID':dataset_test.ID,'target':y_pred})

# prepare the csv file
predictions.to_csv('prediction.csv',index=False)

run_time = time.clock()-start_time
print('Total run time = ' + str(run_time))
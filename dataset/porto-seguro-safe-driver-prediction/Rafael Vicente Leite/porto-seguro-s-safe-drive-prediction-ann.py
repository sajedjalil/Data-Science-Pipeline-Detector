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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 08:55:38 2018

@author: rafaelleite
"""

# Importing the libraries
import matplotlib.pyplot as plt
    
# Importing the datasets
dataset_train = pd.read_csv('../input/train.csv')
dataset_test = pd.read_csv('../input/test.csv')
dataset = pd.concat([dataset_train, dataset_test])

""" In the train and test data, features that belong to similar groupings are tagged 
as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names 
include the postfix bin to indicate binary features and cat to indicate categorical features. 
Features without these designations are either continuous or ordinal. 
Values of -1 indicate that the feature was missing from the observation. 
The target columns signifies whether or not a claim was filed for that policy holder."""

# Replace -1 values with nan on dataset
dataset = dataset.replace(to_replace = -1, value = np.NaN)

# split dataset in X, y
X = dataset.iloc[:, 1:-1]
y = pd.DataFrame(dataset.iloc[:, 58])

# Filling the null values
#dataset.info()
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)
X = pd.DataFrame(data = X, columns = dataset.iloc[:, 1:-1].columns)
#X.info()
#X.dtypes

# setting dtype to category for categorical data
cols = [col for col in X.columns if '_cat' in col]
for col in X[cols]:
    X[col] = X[col].astype('category')
#X.dtypes

# check how many columns we have and delete irrelevant data
ps_car_11_cat = pd.DataFrame(data = [X['ps_car_11_cat'].values])
ps_car_11_cat.apply(pd.value_counts)
cols = [col for col in X.columns if 'ps_car_11_cat' in col]
X = X.drop(columns=cols)

# for the categorical features, we will add dummies
X = pd.get_dummies(X, drop_first = True)

# splitting into test and train datasets
X_train = X.iloc[:595212, :]
X_test = X.iloc[595212:, :]
y_train = y.iloc[:595212, :]
y_test = y.iloc[595212:, :]

# Apply the random under-sampling for X_train
from imblearn.under_sampling import RandomUnderSampler
imbalance_checker = y.apply(pd.value_counts)
under_ratio = 1 - imbalance_checker.target[1]/imbalance_checker.target[0]
rus = RandomUnderSampler(ratio = under_ratio, random_state=42)
X_train_resampled, y_train_resampled = rus.fit_sample(X_train, y_train.values.ravel())
X_train_resampled = pd.DataFrame(X_train_resampled, columns = X_train.columns)
y_train_resampled = pd.DataFrame(y_train_resampled, columns = y_train.columns)
imbalance_checker = y_train_resampled.apply(pd.value_counts)

# for the continuous or ordinal features, we will use feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
cols = [col for col in X_train_resampled.columns if not '_bin' in col and not 'cat' in col]
X_train_resampled[cols] = sc.fit_transform(X_train_resampled[cols])
X_train[cols] = sc.transform(X_train[cols])
X_test[cols] = sc.transform(X_test[cols])


# Backward Eliminiation

# Insert B Intercept
X_train_resampled['constant'] = 1
X_train_resampled = X_train_resampled[['constant'] + X_train_resampled.columns[:-1].tolist()]

# Call Ordinary Least Square and Clean irrelevant data from X_train (except dummies)
import statsmodels.formula.api as sm
xelimination = X_train_resampled
regressorOLS = sm.OLS(y_train_resampled.values.ravel(), xelimination).fit()
regressorOLS.summary()

deleted_columns = []
deleted_pvalues = []

max_column = len(X_train_resampled.columns)

for k in range(0,10):
    for i in range (0 , max_column):
        xelimination = X_train_resampled
        regressorOLS = sm.OLS(y_train_resampled.values.ravel(), xelimination).fit()
    
        pvalue = []
    
        if i < (max_column -1):
            for j in range (1,(max_column - i)):
                pvalue.append(regressorOLS.pvalues[j])
    
            if (max(pvalue) > 0.05):
                index = np.argmax(pvalue) + 1
                if (X_train_resampled.columns[index] != 'constant'):
                    deleted_columns.append(X_train_resampled.columns[index])
                    deleted_pvalues.append(max(pvalue))
                    X_train_resampled.drop(X_train_resampled.columns[index], axis=1, inplace=True)
                    i = 0
                    max_column = max_column - 1
                else:
                    print(index)
                    break
            else:
                break
        

regressorOLS.summary()
regressorOLS.pvalues
X_train_resampled.drop(['constant'], axis=1, inplace=True)
X_train.drop(deleted_columns, axis = 1, inplace = True)
X_test.drop(deleted_columns, axis = 1, inplace = True)

# Splitting Train dataset in N random sample datasets (R% ratio) for training N different ANN's
from random import randrange, uniform
number_of_ann_s = 50
sample_ratio = 1
list_of_dataframes = []

for i in range (0,number_of_ann_s):
    list_of_dataframes.append(pd.DataFrame())

X_train_resampled_y_train_resampled = pd.DataFrame()
X_train_resampled_y_train_resampled = X_train_resampled.join(y_train_resampled)

X_train_resampled_fraction = []
y_train_resampled_fraction = []

X_train_resampled_fraction = list(list_of_dataframes)
y_train_resampled_fraction = list(list_of_dataframes)

for i in range (0,number_of_ann_s):
    X_train_resampled_fraction[i] = pd.DataFrame(data = X_train_resampled_y_train_resampled.sample(frac = uniform(0.5,1) * sample_ratio))
    y_train_resampled_fraction[i] = X_train_resampled_fraction[i].iloc[:,39]
    X_train_resampled_fraction[i] = X_train_resampled_fraction[i].iloc[:,:-1]
    
# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

y_pred_train = []
y_pred_train = list(list_of_dataframes )
y_pred_test = []
y_pred_test = list(list_of_dataframes)


for i in range (0,number_of_ann_s):    
    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense((5 * randrange(1,3)), kernel_initializer = 'uniform', activation = 'relu', input_dim = len(X_train_resampled.columns)))
    
    # Adding the output layer
    classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Fitting the ANN to the Training set
    classifier.fit(X_train_resampled_fraction[i], y_train_resampled_fraction[i].values.ravel(), batch_size = 32, epochs = 110)

    # evaluate the model
    #scores = classifier.evaluate(X_train, y_train)
    #print("\n%s: %.2f%%, ann:%i" % (classifier.metrics_names[1], scores[1]*100, i))
    print("i:%s", i)
    
    # Make prediction
    y_pred_train[i] = classifier.predict(X_train)
    y_pred_test[i] = classifier.predict(X_test)

"""# Simple implementation of the (normalized) gini score in numpy
# Fully vectorized, no python loops, zips, etc.
# Significantly (>30x) faster than previous implementions
def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1/n_samples, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true

y_true = y_train.values.ravel().astype('float32') 
for i in range (0,number_of_ann_s):
    print("%s: %s" % (i, Gini(y_true, y_pred_train[i].reshape(595212))))"""

# Predicting the Test set results
y_pred = pd.DataFrame(columns = ['id', 'target'])
y_pred['id'] = dataset.iloc[595212:, 0].values
y_pred['target'] = np.mean(y_pred_test,axis=0)
y_pred['target'].min()
y_pred['target'].max()


#Part 3 - Generating Submission File
y_pred.to_csv('porto_seguro_submission.csv', index = False)


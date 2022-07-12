# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 08:56:57 2017

@author: ALA
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


# Importing the dataset
dataset = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#Data cleaning
#Remove the unnecesary column for ID

dataset = dataset.drop('Id', axis=1)
test = test.drop('Id', axis=1)

#Remove Constant columns
rm_cons = []
for col in dataset.columns:
    if dataset[col].std() == 0:
        rm_cons.append(col)

print (rm_cons)

dataset.drop(rm_cons, axis =1, inplace = True)
test.drop(rm_cons, axis =1, inplace = True)

#First to second last dataset
X = dataset.iloc[:, :-1].values
#Last column
y = dataset.iloc[:, -1].values

#Test set values in same format as trainng set for xgboost
test_values = test.iloc[:,:].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

 
#--------------------ML Part Begins Here-------------#

#xgb_classifier = xgb.XGBClassifier(missing=np.nan, max_depth=7, n_estimators= 350, learning_rate =0.03, nthread=4, subsample = 0.95, colsample_bytree = 0.85, seed =4242)
#Parameter tuning
xgb_classifier = xgb.XGBClassifier(max_depth=7, n_estimators=350, learning_rate=0.03)
xgb_classifier.fit(X_train, y_train)


#xgb_classifier.fit(X_train, y_train)
y_pred_X_test = xgb_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_X_test)


#Cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=xgb_classifier, X = X_train, y = y_train, cv =10)
print(accuracies.mean())
print(accuracies.std())

#Retrain the model on the whole training set
xgb_classifier.fit(X,y)

#Predict on actual test set
y_pred = xgb_classifier.predict(test_values)
#Submit to Kaggle (need to read test file again to get header info)
test_0 = pd.read_csv('../input/test.csv')

submission=pd.DataFrame({
        "Id": test_0['Id'],
        "Label": y_pred
    })
submission.to_csv("Forest_XGBoost-a_AL.csv", index=False, header=True)


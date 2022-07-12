# -*- coding: utf-8 -*-
"""
Title: "Talking Data Fraud Analysis using XGBoost - Parameter tuning and random sampling"

Goal: Predict whether a user will dowload an app after clicking on an ad 

Created on Tue Apr 24 19:37:12 2018
@author: Julien

"""

# Import basic libraries
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt

# Import Machine learning libraries
import xgboost as xgb
from xgboost import plot_importance


# Run this bloc only the first time
# Create a training dataset by performing random sampling on train.csv
"""
print("\n### Load the data and create a training dataset sample from train.csv: ###")
# Load test.csv
# Build a dataset of 5 million rows selected randomly from train.csv to improve accuracy
train = "train.csv"
# Count the lines
num_lines = sum(1 for l in open(train))
# Sample size
size = 5000000 # you can change it to size = 100000 for the first execution to try it
# The row indices to skip
skip_n = random.sample(range(1, num_lines), num_lines - size)
# Read train.csv
train = pd.read_csv(train, skiprows=skip_n)
"""

print("\n### Load the data: ###")
train = pd.read_csv('../input/train_sample.csv')
test = pd.read_csv('../input/test.csv')

# Display top rows of both train and test datasets to see the features
print(train.head())
print(test.head())

# Compute Descriptive statistics for each variable
print(train.describe())
print(test.describe())

# Feature engineering
print("\n### Feature engineering: ###")
# Count the number of clicks per ip address
print("Number of Clicks per ip:")

nb_clicks = train.groupby('ip')
train['nb_clicks'] = nb_clicks['ip'].transform(lambda x : x.count())

nb_clicks = test.groupby('ip')
test['nb_clicks'] = nb_clicks['ip'].transform(lambda x : x.count())

# Extract day and hour from 'click_time'
train['day'] = pd.to_datetime(train.click_time).dt.day.astype('uint8')
train['hour'] = pd.to_datetime(train.click_time).dt.hour.astype('uint8')

test['day'] = pd.to_datetime(test.click_time).dt.day.astype('uint8')
test['hour'] = pd.to_datetime(test.click_time).dt.hour.astype('uint8')

#You can save these dataframes to csv files in order to faster next excutions
#train.to_csv("train2.csv",header=True,index=False)
#test.to_csv("test2.csv",header=True,index=False)

print("\n### Data preaparation before training: ###")
# Extract X_train and Y_train from train
X_train = train.drop("is_attributed",axis=1)
X_train = X_train.drop("click_time",axis=1)
X_train = X_train.drop("attributed_time",axis=1)
Y_train = train["is_attributed"]

# Build X_pred from test (we want to predic Y_pred from X_pred features)
X_pred  = test.drop("click_time",axis=1)
X_pred = X_pred.drop("click_id",axis=1)

# We ensure that X_train and X_pred have the same shape before training the data
print(X_train.shape)
print(X_pred.shape)
print(Y_train.shape)


print("\n### Cross validation splitting: ###")
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size = 0.20, random_state = 0)
eval_set = [(x_train, y_train), (x_test,y_test)]

print("\n### Creating XGBoost Clasifier: ###")
# XGBoost - parameters have been tuned
xgboost = xgb.XGBClassifier(objective="binary:logistic",min_child_weight=1, max_depth=5, learning_rate=.1, 
                            n_estimators=1000, n_jobs=-1,gamma=0, nthread=3, scale_pos_weight=1, seed=27)


print("\n### Fit the model: ###")
# eval_metric="auc" -> Area Under the Curve      
# early_stopping_rounds is the maximum number of iterations executed without improvmen of eval_metric
xgboost.fit(x_train,y_train,eval_set=eval_set, eval_metric="auc", early_stopping_rounds=25)

#Accuracy

print("\n### Accuracy: ###")
#y_pred = xgboost.predict(X_pred)
#print(y_pred)
y_pred = xgboost.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_test)
print(accuracy)


# Feature importance
# Plot feature importance
plot_importance(xgboost)
plt.show()


#Confusion matrix in order to help you tune your model
print("Confusion matrix")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Compute False Positive and False negative

# True Positive 
TP = np.sum(np.logical_and(y_pred == 1, y_test == 1))
# True Negative
TN = np.sum(np.logical_and(y_pred == 0, y_test == 0))
# False Positive
FP = np.sum(np.logical_and(y_pred == 1, y_test == 0))
 # False Negative
FN = np.sum(np.logical_and(y_pred == 0, y_test == 1))

print("False positive:",FP/TP)
print("False Negative:",FN/TN)
print("Sum false:",FP+FN)

print("\n### Compute prediction: ###")

y_pred = xgboost.predict_proba(X_pred)

print("Export...")
submission = pd.DataFrame({
        "click_id": test["click_id"],
        "is_attributed": y_pred[:,1]
    })

print("\n### Create submission file: ###")
submission.to_csv('ads_fraud_sub.csv', index=False)
print('Exported')

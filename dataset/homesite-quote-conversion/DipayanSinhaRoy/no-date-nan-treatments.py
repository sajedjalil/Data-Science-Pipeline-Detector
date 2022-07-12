# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------
Kaggle Homesite Insurance Competition
Author: Raj Saha, Dipayan Sinha Roy
Team Sukhen
-----------------------------------------------------------------------------
"""
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest

#os.chdir("C:\SparkCourse\kaggleinsurance")

"""
Data Engineering and Analysis
"""
#Load the dataset

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

#train_data = pd.read_csv("train.csv")
#test_data = pd.read_csv("test.csv")


"""
Data Transformations

Let us do the following transformations

1. Convert Date into separate columns - year, month, week
2. Convert all non numeric data to numeric
"""


# Converting date into datetime format
train_data['Date'] = pd.to_datetime(pd.Series(train_data['Original_Quote_Date']))
# Dropping original date column
del train_data['Original_Quote_Date']
## Seperating date into 3 columns
train_data['Month'] = train_data['Date'].apply(lambda x: int(str(x)[5:7]))
train_data['weekday'] = train_data['Date'].dt.dayofweek
train_data['Year'] = train_data['Date'].apply(lambda x: int(str(x)[:4]))




# Converting date into datetime format
test_data['Date'] = pd.to_datetime(pd.Series(test_data['Original_Quote_Date']))
# Dropping original date column
del test_data['Original_Quote_Date']
## Seperating date into 3 columns
test_data['Month'] = test_data['Date'].apply(lambda x: int(str(x)[5:7]))
test_data['weekday'] = test_data['Date'].dt.dayofweek
test_data['Year'] = test_data['Date'].apply(lambda x: int(str(x)[:4]))



del train_data['Date']
del test_data['Date']

#data imputation    
train_data['PersonalField7']=train_data['PersonalField7'].fillna('N')
del train_data['PersonalField84']
train_data['PropertyField3']=train_data['PropertyField3'].fillna('N')
train_data['PropertyField4']=train_data['PropertyField4'].fillna('N')
del train_data['PropertyField29']
train_data['PropertyField32']=train_data['PropertyField32'].fillna('Y')
train_data['PropertyField34']=train_data['PropertyField34'].fillna('Y')
train_data['PropertyField36']=train_data['PropertyField36'].fillna('N')
train_data['PropertyField38']=train_data['PropertyField38'].fillna('N')

test_data['PersonalField7']=test_data['PersonalField7'].fillna('N')
del test_data['PersonalField84']
test_data['PropertyField3']=test_data['PropertyField3'].fillna('N')
test_data['PropertyField4']=test_data['PropertyField4'].fillna('N')
del test_data['PropertyField29']
test_data['PropertyField32']=test_data['PropertyField32'].fillna('Y')
test_data['PropertyField34']=test_data['PropertyField34'].fillna('Y')
test_data['PropertyField36']=test_data['PropertyField36'].fillna('N')
test_data['PropertyField38']=test_data['PropertyField38'].fillna('N')

train_data=train_data.fillna(-1)
test_data=test_data.fillna(-1)

#Convert all strings to equivalent numeric representations
#to do correlation analysis
for f in train_data.columns:
    if train_data[f].dtype=='object':
        print(f)
        lbl=preprocessing.LabelEncoder()
        lbl.fit(list(train_data[f].values)+list(test_data[f].values))
        train_data[f]=lbl.transform(list(train_data[f].values))
        test_data[f]=lbl.transform(list(test_data[f].values))
    

train_data['y']=train_data['QuoteConversion_Flag']
del train_data['QuoteConversion_Flag']

"""
Modeling and Prediction
"""

predictors1 = train_data.iloc[:,1:295]
targets = train_data.y

pred_test1=test_data.iloc[:,1:295]
                 
from sklearn.feature_selection import SelectKBest, chi2 , f_classif
fc = SelectKBest(f_classif, k=260)
predictors = fc.fit_transform(predictors1, targets)
pred_test = fc.transform(pred_test1)

import xgboost as xgb
train_xgb=xgb.DMatrix(predictors, targets)
test_xgb=xgb.DMatrix(pred_test) 
#params = {"objective": "binary:logistic"}
params = { "objective": "binary:logistic"}
gbm = xgb.train(params, train_xgb, 40)
predictions_xgb = gbm.predict(test_xgb)


"""
Saving the results in submission file
"""
test_data["QuoteConversion_Flag"]=predictions_xgb
test_data[["QuoteNumber","QuoteConversion_Flag"]].to_csv("xgboost_nodate_nan.csv", index=False)
                




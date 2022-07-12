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



"""
Data Engineering and Analysis
"""
#Load the dataset

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

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
train_data['Year'] = train_data['Date'].apply(lambda x: int(str(x)[:4]))
train_data['Month'] = train_data['Date'].apply(lambda x: int(str(x)[5:7]))
train_data['weekday'] = train_data['Date'].dt.dayofweek


# Converting date into datetime format
test_data['Date'] = pd.to_datetime(pd.Series(test_data['Original_Quote_Date']))
# Dropping original date column
del test_data['Original_Quote_Date']
## Seperating date into 3 columns
test_data['Year'] = test_data['Date'].apply(lambda x: int(str(x)[:4]))
test_data['Month'] = test_data['Date'].apply(lambda x: int(str(x)[5:7]))
test_data['weekday'] = test_data['Date'].dt.dayofweek

del train_data['Date']
del test_data['Date']



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

predictors = train_data.iloc[:,1:300]
targets = train_data.y

pred_test=test_data.iloc[:,1:300]
                 

import xgboost as xgb
train_xgb=xgb.DMatrix(predictors, targets)
test_xgb=xgb.DMatrix(pred_test) 
params = {"objective": "binary:logistic"}

gbm = xgb.train(params, train_xgb, 20)
predictions_xgb = gbm.predict(test_xgb)


"""
Saving the results in submission file
"""
test_data["QuoteConversion_Flag"]=predictions_xgb
test_data[["QuoteNumber","QuoteConversion_Flag"]].to_csv("take6_xgboost.csv", index=False)
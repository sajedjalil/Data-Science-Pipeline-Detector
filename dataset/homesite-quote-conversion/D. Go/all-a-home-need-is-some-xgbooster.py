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
train_data['weekday'] = train_data['Date'].dt.dayofweek
train_data['Year'] = train_data['Date'].apply(lambda x: int(str(x)[:4]))
train_data['Month'] = train_data['Date'].apply(lambda x: int(str(x)[5:7]))



# Converting date into datetime format
test_data['Date'] = pd.to_datetime(pd.Series(test_data['Original_Quote_Date']))
# Dropping original date column
del test_data['Original_Quote_Date']
## Seperating date into 3 columns
test_data['weekday'] = test_data['Date'].dt.dayofweek
test_data['Year'] = test_data['Date'].apply(lambda x: int(str(x)[:4]))
test_data['Month'] = test_data['Date'].apply(lambda x: int(str(x)[5:7]))

train_data['Custom_pers_fiel_10a_na'] = train_data['PersonalField10A'].apply(lambda x: 0 if(x == -1) else 1)
test_data['Custom_pers_fiel_10a_na']  =  test_data['PersonalField10A'].apply(lambda x: 0 if(x == -1) else 1)

del train_data['Date']
del test_data['Date']

train_data['PersonalField7']=train_data['PersonalField7'].fillna('N')
#del train_data['PersonalField84']
train_data['PropertyField3']=train_data['PropertyField3'].fillna('N')
train_data['PropertyField4']=train_data['PropertyField4'].fillna('N')
#del train_data['PropertyField29']
train_data['PropertyField32']=train_data['PropertyField32'].fillna('Y')
train_data['PropertyField34']=train_data['PropertyField34'].fillna('Y')
train_data['PropertyField36']=train_data['PropertyField36'].fillna('N')
train_data['PropertyField38']=train_data['PropertyField38'].fillna('N')

test_data['PersonalField7']=test_data['PersonalField7'].fillna('N')
#del test_data['PersonalField84']
test_data['PropertyField3']=test_data['PropertyField3'].fillna('N')
test_data['PropertyField4']=test_data['PropertyField4'].fillna('N')
#del test_data['PropertyField29']
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

predictors1 = train_data.iloc[:,1:298]
targets = train_data.y

pred_test1=test_data.iloc[:,1:298]
from sklearn.feature_selection import SelectKBest, chi2 , f_classif
fc = SelectKBest(f_classif, k=290)
predictors = fc.fit_transform(predictors1, targets)
pred_test = fc.transform(pred_test1)
#w = { 0 : 2 , 1 : 1 }
import xgboost as xgb
train_xgb=xgb.DMatrix(predictors, targets)
test_xgb=xgb.DMatrix(pred_test) 
params = {"max_depth": 4 , "objective": "binary:logistic"}
#params = {'bst:max_depth':15,  'objective':'binary:logistic' }

# 75  -> 96502
# 150 -> 96650
# 200 -> 
gbm = xgb.train(params, train_xgb, 75)
predictions_xgb = gbm.predict(test_xgb)


num_round = 10

#print ('running cross validation')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
#print(xgb.cv(params, train_xgb, num_round, nfold=2,metrics={'auc'}, seed = 0))
"""
Saving the results in submission file
"""
test_data["QuoteConversion_Flag"]=predictions_xgb
test_data[["QuoteNumber","QuoteConversion_Flag"]].to_csv("take6_xgboost.csv", index=False)
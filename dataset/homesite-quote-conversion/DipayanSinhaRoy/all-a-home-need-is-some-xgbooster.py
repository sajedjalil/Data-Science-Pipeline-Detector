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
#params = {"max_depth": 4 , "objective": "binary:logistic"}

seed = 1718
params =     {
    #1- General Parameters       
    'booster' : "gbtree", #booster [default=gbtree]
    'silent': 0 , #silent [default=0]
    #'nthread' : -1 , #nthread [default to maximum number of threads available if not set]

    #2A-Parameters for Tree Booster   
    #'eta'  :0.023, # eta [default=0.3] range: [0,1]
    #'gamma':0 ,#gamma [default=0] range: [0,âˆž]
    'max_depth'           :6, #max_depth [default=6] range: [1,âˆž]
    #'min_child_weight':1,  #default=1]range: [0,âˆž]
    #'max_delta_step':0, #max_delta_step [default=0] range: [0,âˆž]
    'subsample'           :0.9, #subsample [default=1]range: (0,1]
    'colsample_bytree'    :0.85, #colsample_bytree [default=1]range: (0,1]
    #'lambda': 1,  #lambda [default=1]
    #'alpha':0.0001, #alpha [default=0]
    
    
    #2B- Parameters for Linear Booster
    #'lambda': 0,  #lambda [default=0]
    #'alpha':0, #alpha [default=0]
    #'lambda_bias':0, #default 0
    
    #3- earning Task Parameters
    'objective': 'binary:logistic',  #objective [ default=reg:linear ]
    #'base_score'=0.5,        #base_score [ default=0.5 ]
    'eval_metric' : 'auc', #eval_metric [ default according to objective ]
    'seed':0 #seed #seed [ default=0 ]
  
    }
#params = {'bst:max_depth':15,  'objective':'binary:logistic' }
gbm = xgb.train(params, train_xgb, 125)
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
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



train_data=train_data.fillna(-999)
test_data=test_data.fillna(-999)

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
fc = SelectKBest(f_classif, k=285)
predictors = fc.fit_transform(predictors1, targets)
pred_test = fc.transform(pred_test1)
#w = { 0 : 2 , 1 : 1 }
import xgboost as xgb
train_xgb=xgb.DMatrix(predictors, targets,missing=-999)
#test_xgb=xgb.DMatrix(pred_test) 
#params = {"objective": "binary:logistic"}
params = {'bst:max_depth':15, 'bst:eta':.1, 'objective':'binary:logistic' }
#gbm = xgb.train(params, train_xgb, 40)
#predictions_xgb = gbm.predict(test_xgb)


num_round = 10

print ('running cross validation')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
print(xgb.cv(params, train_xgb, num_round, nfold=2,metrics={'auc'}, seed = 0))
"""
Saving the results in submission file
"""
#test_data["QuoteConversion_Flag"]=predictions_xgb
#test_data[["QuoteNumber","QuoteConversion_Flag"]].to_csv("take6_xgboost.csv", index=False)
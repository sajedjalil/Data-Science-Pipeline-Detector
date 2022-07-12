# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 18:09:17 2020

@author: Vithal Nistala
"""

import xgboost as xgb 
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
dt = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
df['Target'].replace(['ConfirmedCases','Fatalities'],[0,1],inplace=True)
dt['Target'].replace(['ConfirmedCases','Fatalities'],[0,1],inplace=True)

df['date_dup'] = pd.to_datetime(df['Date'])
df['month'] = 0
list1=[]
for i in df['date_dup']:
    list1.append(i.month)
df['month'] = list1
df['date'] = 0
list1=[]
for i in df['date_dup']:
    list1.append(i.day)
df['date'] = list1


dt['date_dup'] = pd.to_datetime(dt['Date'])
dt['month'] = 0
list1=[]
for i in dt['date_dup']:
    list1.append(i.month)
dt['month'] = list1
dt['date'] = 0
list1=[]
for i in dt['date_dup']:
    list1.append(i.day)
dt['date'] = list1

#dt['ForecastId'] =dt['Id']
dt.rename(columns={"ForecastId": "Id"},inplace = True) 
df.drop(['County','Province_State','Country_Region','Date','date_dup'],axis = 1,inplace = True)
dt.drop(['County','Province_State','Country_Region','Date','date_dup'],axis = 1,inplace = True)


X=df.drop(['TargetValue'],axis=1)
y=df['TargetValue']

x_test = dt



# Instantiate the XGBRegressor: xg_reg
xg_reg = xgb.XGBRegressor(objective='reg:linear',n_estimators = 10,seed=123)

# Fit the regressor to the training set
xg_reg.fit(X,y)

# Predict the labels of the test set: preds
y_pred = xg_reg.predict(x_test)



pred=pd.DataFrame(y_pred)
#print(pred)
sub_df=pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')
#print(sub_df.shape)
datasets=pd.concat([sub_df['ForecastId_Quantile'],pred],axis=1)
datasets.columns=['ForecastId_Quantile','TargetValue']
datasets.to_csv('samplesubmission.csv',index=False)

                                                    

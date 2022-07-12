# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 18:09:17 2020

@author: Vithal Nistala
"""

#import xgboost as xgb 
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


df.drop(['County','Province_State','Country_Region','Date','date_dup'],axis = 1,inplace = True)
dt.drop(['County','Province_State','Country_Region','Date','date_dup'],axis = 1,inplace = True)


X=df.drop(['TargetValue'],axis=1)
y=df['TargetValue']

x_test = dt


rfr = RandomForestRegressor()
rfr.fit(X,y)
y_pred = rfr.predict(x_test)

pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('submission.csv')
#print(sub_df.shape)
datasets=pd.concat([sub_df['ForecastId_Quantile'],pred],axis=1)
datasets.columns=['ForecastId_Quantile','TargetValue']
datasets.to_csv('samplesubmission.csv',index=False)

                                                    

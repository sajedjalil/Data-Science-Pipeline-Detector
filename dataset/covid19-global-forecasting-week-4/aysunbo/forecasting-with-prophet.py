# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import sys

import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn
import fbprophet
from fbprophet import Prophet

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Read data
test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
#prepare test set
d_test = test.drop(test.loc[:, 'ForecastId':'Country_Region'].columns, axis = 1) 
d_test = d_test.drop_duplicates(subset=None, keep='first', inplace=False)
d_test.rename(columns = {'Date':'ds'}, inplace=True)
#train set for the prediction of confirmed cases
train_conf = train.drop(['Id', 'Fatalities','Province_State'], axis = 1) 
#train period for each country
train_period = len(train_conf['Date'].unique())
#train data and predict confirmed cases for each country using Prophet
prediction_list=[]
count = train_period
start=0
while count<=len(train_conf):
  train_country = train_conf.loc[start:count,['Date','ConfirmedCases']]
  train_country.rename(columns = {'Date':'ds', 'ConfirmedCases':'y'}, inplace=True)
  model = Prophet(weekly_seasonality=True)
  model.add_seasonality(name='monthly', period = 30.3, fourier_order= 5)
  model.fit(train_country)
  predictions=model.predict(d_test)
  prediction_list.append(predictions['yhat'])
  start = count
  count+=train_period

#train set for the prediction of fatalities in each country
train_fat = train.drop(['Id', 'ConfirmedCases','Province_State', 'Country_Region'], axis = 1) 
#train data and predict fatalities for each country using Prophet
prediction_list_fat=[]
count = train_period
start=0
while count<=len(train_fat):
  train_country_fat = train_fat.loc[start:count,['Date','Fatalities']]
  train_country_fat.rename(columns = {'Date':'ds', 'Fatalities':'y'}, inplace=True)
  model = Prophet(weekly_seasonality=True)
  model.add_seasonality(name='monthly', period = 30.3, fourier_order= 5)
  model.fit(train_country_fat)
  predictions_fat=model.predict(d_test)
  prediction_list_fat.append(predictions_fat['yhat'])
  start = count
  count+=train_period

#preparing submission file
submission_df = pd.DataFrame(columns=["ForecastId","ConfirmedCases","Fatalities"])
submission_df['ForecastId']=test['ForecastId']
count = 0
for i in range(len(prediction_list)):
  for j in range(len(d_test)):
    value_conf = float(str(prediction_list[i][j]).split('\t')[-1])
    value_fat = float(str(prediction_list_fat[i][j]).split('\t')[-1])
    submission_df.loc[count,['ConfirmedCases']]=value_conf
    submission_df.loc[count,['Fatalities']]=value_fat
    count+=1
submission_df.to_csv("submission.csv",index=False)

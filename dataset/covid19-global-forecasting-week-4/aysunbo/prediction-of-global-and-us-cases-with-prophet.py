# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
#prediction of global confirmed cases
#aggregate cases and prepare data for Prophet algorithm
daily_confirmed = train.groupby("Date")[['ConfirmedCases']].sum()
daily_confirmed.describe()
daily_confirmed.plot(kind = 'line', figsize = (10,5))
train_daily_confirmed = daily_confirmed.reset_index()
train_daily_confirmed.rename(columns = {'Date':'ds', 'ConfirmedCases':'y'}, inplace=True)
#Prophet model 
model = Prophet(weekly_seasonality=True)
model.add_seasonality(name='monthly', period = 30.3, fourier_order= 5)
model.fit(train_daily_confirmed)
prediction = model.predict(d_test)
#Evaluation
overlap = train_daily_confirmed.iloc [71:,1] 
overlap = overlap.reset_index()
comparison = prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
comparison['y_actual'] = overlap.iloc[:,1]
comparison ['error'] = comparison['yhat'] -  comparison['y_actual']
comparison ['abs_error']= abs(comparison['error'])
print(comparison)
print('mean absolute error', comparison ['abs_error'].sum()/13)
#Plot
model.plot(prediction)
model.plot_components(prediction)

#prediction of global fatalities
#prepare data
daily_fatalities = train.groupby("Date")[['Fatalities']].sum()
daily_fatalities.plot(kind = 'line', figsize = (10,5))
train_daily_fatalities = daily_fatalities.reset_index()
train_daily_fatalities.rename(columns = {'Date':'ds', 'Fatalities':'y'}, inplace=True)
#Prophet Model for fatalities
model_for_fatalities = Prophet(weekly_seasonality=True)
model_for_fatalities.add_seasonality(name='monthly', period = 30.3, fourier_order= 5)
model_for_fatalities.fit(train_daily_fatalities)
predicted_fatalities = model_for_fatalities.predict(d_test)
#Evaluation
overlap_fatalities = train_daily_fatalities.iloc [71:,1] 
overlap_fatalities = overlap_fatalities.reset_index()
comparison_fatalities = predicted_fatalities[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
comparison_fatalities ['y_actual'] = overlap_fatalities.iloc[:,1]
comparison_fatalities ['error'] = comparison_fatalities['yhat'] -  comparison_fatalities['y_actual']
comparison_fatalities ['abs_error']= abs(comparison_fatalities['error'])
print(comparison_fatalities)
print('mean absolute error', comparison_fatalities ['abs_error'].sum()/13)
#Plots 
model_for_fatalities.plot(predicted_fatalities)
model_for_fatalities.plot_components(predicted_fatalities)

#Prediction of Confirmed Cases in US
#prepare data
US = train['Country_Region'] == "US"
US_confirmed = train[US].groupby("Date")[['ConfirmedCases']].sum()
train_US_confirmed = US_confirmed.reset_index()
train_US_confirmed.rename(columns = {'Date':'ds', 'ConfirmedCases':'y'}, inplace=True)
#Prophet Model for US
model_for_US = Prophet(weekly_seasonality=True)
model_for_US.add_seasonality(name='monthly', period = 30.3, fourier_order= 5)
model_for_US.fit(train_US_confirmed)
US_confirmed_predicted = model_for_US.predict(d_test)
#Evaluation
overlap_US = train_US_confirmed.iloc [71:,1] 
overlap_US = overlap_US.reset_index()
comparison_US = US_confirmed_predicted [['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
comparison_US ['y_actual'] = overlap_US.iloc[:,1]
comparison_US ['error'] = comparison_US['yhat'] -  comparison_US['y_actual']
comparison_US ['abs_error']= abs(comparison_US['error'])
print(comparison_US)
print('mean absolute error', comparison_US ['abs_error'].sum()/13)
#Plots
model_for_US.plot(US_confirmed_predicted)
model_for_US.plot_components(US_confirmed_predicted)

#Prediction of Fatalities in US
#prepare data
US_fatalities = train[US].groupby("Date")[['Fatalities']].sum()
train_US_fatalities = US_fatalities.reset_index()
train_US_fatalities.rename(columns = {'Date':'ds', 'Fatalities':'y'}, inplace=True)
#Prophet Model for Fatalities in US
model_for_US_f = Prophet(weekly_seasonality=True)
model_for_US_f.add_seasonality(name='monthly', period = 30.3, fourier_order= 5)
model_for_US_f.fit(train_US_fatalities)
US_fatalities_predicted = model_for_US_f.predict(d_test)
#Evaluation
overlap_US_fatalities = train_US_fatalities.iloc [71:,1] 
overlap_US_fatalities = overlap_US_fatalities.reset_index()
comparison_US_fatalities = US_fatalities_predicted [['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
comparison_US_fatalities ['y_actual'] = overlap_US_fatalities.iloc[:,1]
comparison_US_fatalities ['error'] = comparison_US_fatalities['yhat'] -  comparison_US_fatalities['y_actual']
comparison_US_fatalities ['abs_error']= abs(comparison_US_fatalities['error'])
print(comparison_US_fatalities)
print('mean absolute error', comparison_US_fatalities ['abs_error'].sum()/13)
#Plot
model_for_US_f.plot(US_fatalities_predicted)
model_for_US_f.plot_components(US_fatalities_predicted)


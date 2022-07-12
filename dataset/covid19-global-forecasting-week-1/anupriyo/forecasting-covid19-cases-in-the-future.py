# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as py
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
train.columns
train.head()
train.tail()
train.isna().sum()
test1=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
test1.head()
train_group=train.groupby('Country/Region')[['Date','ConfirmedCases','Fatalities']].sum().reset_index()
train_group
train_date=train.groupby('Date')[['ConfirmedCases','Fatalities']].sum().reset_index()
train_date['Date']=train_date['Date'].astype('datetime64')
train_date.tail()
train_date.dtypes
plt.plot(train_date.Date,train_date.ConfirmedCases,label='ConfirmedCases')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.plot(train_date.Date,train_date.Fatalities,label='Fatalities')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Fatalities')
#Forecasting Confirmed Case
train_confirmed=train_date[['Date','ConfirmedCases']]
train_confirmed.columns=['ds','y']
m=Prophet()
m.fit(train_confirmed)
future=m.make_future_dataframe(periods=5*7)
forecast_test=m.predict(future)
forecast_test
test=forecast_test[['ds','trend']]
test=test[test['trend']>0]
test.columns = ['Date','ConfirmedCases']
test.tail()
test.head(30)
fig_test = plot_plotly(m, forecast_test)
py.iplot(fig_test) 
fig_test = m.plot(forecast_test,xlabel='Date',ylabel='Confirmed Cases')

#Forecasting fatalities 
train_fatalities=train_date[['Date','Fatalities']]
train_fatalities.columns=['ds','y']
m1=Prophet()
m1.fit(train_fatalities)
future=m1.make_future_dataframe(periods=5*7)
forecast_test=m1.predict(future)
forecast_test
test=forecast_test[['ds','trend']]
test=test[test['trend']>0]
test.columns = ['Date','Fatalities']
test.tail()
fig_test = plot_plotly(m1,forecast_test)
py.iplot(fig_test) 
fig_test = m1.plot(forecast_test,xlabel='Date',ylabel='Deaths')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
os.listdir('../input')
# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import pandas as pd
import datetime as dt
import numpy as np
from numpy.random import poisson
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


path = '/kaggle/input/covid19-global-forecasting-week-2/'

def prepare_data(f):
    # read file
    train = pd.read_csv(f)

    # Change colum Date to datetime
    train['Date'] = pd.to_datetime(train['Date'])

    # origine of dates
    date0 = dt.datetime.strptime('2020-01-22', '%Y-%m-%d')

    # Calculate Time column
    train['Time'] = train['Date']-date0
    i = 0
    n = len(train['Time'])
    for i in range(n):
        train.loc[i, 'Time'] = train.loc[i, 'Time'].days

    # fill column "Province_State" with value in  "Country_Region" if empty
    for i in range(n):
        if pd.isnull(train.loc[i, "Province_State"]):
            train.loc[i, "Province_State"] = train.loc[i,
                                                       "Country_Region"] + '_'

    return train



def first_confirmedCase(df, number=1):
    temp = df.loc[df['ConfirmedCases'] >= number]
    return (pd.DataFrame(temp.groupby(["Province_State"])['Time'].min())).reset_index()


def first_fatality(df, number=1):
    temp = df.loc[df['Fatalities'] >= number]
    return pd.DataFrame(temp.groupby(["Province_State"])['Time'].min())


def get_province_state(df, province):
    temp = df.loc[df['Province_State'] == province, :]
    temp = temp.reset_index()
    return temp


def get_country_region(df, country, aggregate=True):
    temp = df.loc[df['Country_Region'] == country, :]

    if aggregate:
        temp = temp.groupby(["Time", "Date"])
        return (pd.DataFrame(temp['ConfirmedCases', 'Fatalities'].sum())).reset_index()
    else:
        return (temp.loc[:, ['Province_State', 'Time', 'Date', 'ConfirmedCases', 'Fatalities']]).reset_index()


def get_province_state_serie(df, province):
    temp = df.loc[df['Province_State'] == province, :]
    temp = temp.reset_index()
    return temp

time_begin = 0


train0 = prepare_data(path + 'train.csv')

test = prepare_data(path + 'test.csv')

test['ConfirmedCases'] = 0

test['Fatalities'] = 0

test = test.set_index(['Province_State', 'Date'])

# populate test with values available in train
temp = train0.set_index(['Province_State', 'Date'])
temp = temp.loc[:, ['ConfirmedCases', 'Fatalities']]
test.update(temp)

provinces = list(train0['Province_State'].unique())

lags = [14, 12, 10, 8, 6, 4, 2]



for province in provinces:
    train = get_province_state(train0, province)

    train = (train.loc[train['Time'] >= time_begin, :]
             ).loc[:, ['Date', 'ConfirmedCases']]

    region_data = train.set_index('Date')


    # fit model
    for l in lags:
        try:
            model = ARIMA(region_data, order=(l, 1, 1), freq='D')
            model_fit = model.fit(disp=0)
            break
        except:
            continue


    prediction = model_fit.predict(
        start=69, end=99, typ='levels')

    prediction_df = pd.DataFrame(prediction, columns=['ConfirmedCases'])

    prediction_df['Province_State'] = province

    prediction_df = prediction_df.reset_index()

    prediction_df = prediction_df.rename(columns={'index': 'Date'})

    prediction_df = prediction_df.set_index(['Province_State', 'Date'])

    test.update(prediction_df)

    
    
for province in provinces:
    train = get_province_state(train0, province)

    train = (train.loc[train['Time'] >= time_begin, :]
             ).loc[:, ['Date', 'Fatalities']]

    region_data = train.set_index('Date')


    # fit model
    for l in lags:
        try:
            model = ARIMA(region_data, order=(l, 1, 1), freq='D')
            model_fit = model.fit(disp=0)
            break
        except:
            continue


    prediction = model_fit.predict(
        start=69, end=99, typ='levels')

    prediction_df = pd.DataFrame(prediction, columns=['Fatalities'])

    prediction_df['Province_State'] = province

    prediction_df = prediction_df.reset_index()

    prediction_df = prediction_df.rename(columns={'index': 'Date'})

    prediction_df = prediction_df.set_index(['Province_State', 'Date'])

    test.update(prediction_df)


test = test.reset_index()

test = test.loc[:,['ForecastId','ConfirmedCases','Fatalities']]

test.to_csv('submission.csv', index=False)
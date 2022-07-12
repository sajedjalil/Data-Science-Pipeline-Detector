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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:50:46 2020

@author: serhiy
"""

import time, sys
import pandas as pd
import numpy as np
from fbprophet import Prophet
from tqdm import tqdm, tnrange
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning, )

future_preds = 28

#%% Read data
df_calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
df_train = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv', index_col=0)
df_train.index = df_train.index.str.replace('_validation', '')
day_cols = pd.Series([c for c in df_train.columns if c.find('d_')==0])

#%% Clean data: remove leading zeros and outliers
def clean_data(df_train, day_cols, indx):
    t = df_train.loc[indx].copy()
    t.loc[day_cols[((t.loc[day_cols]>0).cumsum()==0).values]] = np.nan

    q1 = t.loc[day_cols].quantile(0.25)
    q3 = t.loc[day_cols].quantile(0.75)
    iqr = q3-q1
    qm = (q3+1.5*iqr)
    t.loc[day_cols][t.loc[day_cols]>qm] = qm
    return t

#%% Prepare calendar columns
df_calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
df_calendar.index = df_calendar['d'].values
df_calendar['ds'] = pd.to_datetime(df_calendar['date'])
df_calendar['quarter'] = df_calendar['ds'].dt.quarter

#%% Generate holidays ds
events1 = pd.Series(df_calendar['event_name_1'].values, index=df_calendar['ds'].values).dropna()
events2 = pd.Series(df_calendar['event_name_2'].values, index=df_calendar['ds'].values).dropna()
holidays = pd.DataFrame(pd.concat([events1, events2], axis=0))

holidays['ds'] = holidays.index.values
holidays.rename({0: 'holiday'}, axis=1, inplace=True)
holidays.reset_index(drop=True, inplace=True)
del events1, events2
#%%
print('Cleaning data...', flush=True)
#data = [clean_data(df_train, day_cols, i) for i in tqdm(df_train.index)]
data = [clean_data(df_train, day_cols, i) for i in df_train.index]
df_train = pd.concat(data, axis=1).T

#%% Read and prepare price table
print('Prepare prices')
df_sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
df_sell_prices['id'] = df_sell_prices['item_id'] + '_' + df_sell_prices['store_id']

df_sell_prices = df_sell_prices.pivot(index='id', columns='wm_yr_wk', values='sell_price')
df_sell_prices = df_sell_prices.fillna(method='bfill', axis=1)
df_prices = pd.DataFrame(index=df_train.index.values)
df_prices = list()
for i in df_sell_prices.columns:
    cols = df_calendar['d'][df_calendar['wm_yr_wk']==i]
    t = pd.concat([df_sell_prices[i] for j in cols], axis=1)
    t.columns = cols
    df_prices.append(t)
df_prices = pd.concat(df_prices, axis=1)

#%% make_prediction definition
def make_prediction(indx):
    global df_train, holidays, df_prices, df_calendar
    changepoints=list()
    uncertainty_samples=False
    changepoint_prior_scale=0.1
    changepoint_range=0.9
    holidays_prior_scale=10
    yearly_seasonality=2
    weekly_seasonality=1
    daily_seasonality=False
    monthly_fourier_order=None
    quarter_fourier_order=None
    seasonality_prior_scale=10
    seasonality_mode = 'additive'
    
    target = df_train.loc[indx, day_cols]

    df = df_calendar.iloc[:target.shape[0]+future_preds][['ds', 'month', 'wday', 'quarter', 'snap_'+df_train.loc[indx, 'state_id']]]
    df['y'] = target
    df['prices'] = df_prices.loc[indx].iloc[:df.shape[0]].values

    m = Prophet(growth='linear', uncertainty_samples=uncertainty_samples, changepoint_prior_scale=changepoint_prior_scale, changepoint_range=changepoint_range,
                holidays_prior_scale=holidays_prior_scale, yearly_seasonality=yearly_seasonality,
                daily_seasonality=daily_seasonality, weekly_seasonality=weekly_seasonality,
                holidays=holidays, seasonality_mode=seasonality_mode, seasonality_prior_scale=seasonality_prior_scale)

    if not monthly_fourier_order is None:
        m.add_seasonality(name='monthly', period=365.25/12, fourier_order=monthly_fourier_order)
    if not quarter_fourier_order is None:
        m.add_seasonality(name='quarterly', period=365.25/4, fourier_order=quarter_fourier_order)#, prior_scale=15)

    for reg in df.columns:
        if reg!='ds' and reg!='y':
            m.add_regressor(reg)
    m.fit(df.loc[target.loc[target.first_valid_index():].index])

    df.drop(['y'], axis=1, inplace=True)
    
    forecast = m.predict(df.iloc[-future_preds:])
    res = forecast['yhat']
    res.index = df.iloc[-future_preds:].index.values
    
    return res
#%%
submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv', index_col=0)
#%
print('Predicting...', flush=True)
start_time = time.time()
pool = Pool()

train_indxs = df_train.index

#_, train_indxs = train_test_split(df_train.index, test_size=10, random_state=1246)
#t = make_prediction(train_indxs[0])

res = pool.map(make_prediction, train_indxs)
pool.close()
pool.join()
end_time = time.time()
print('Exec speed=%.2f' %((end_time-start_time)/train_indxs.shape[0]))
#%%
for j, i in enumerate(train_indxs):
    submission.loc[i+'_validation'] = res[j].values
    submission.loc[i+'_evaluation'] = res[j].values
#%%
submission[submission<0]=0
submission.to_csv('submission.csv')
#%%

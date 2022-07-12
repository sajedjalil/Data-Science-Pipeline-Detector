# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd 
from statsmodels.tsa.statespace.sarimax import SARIMAX
#from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot, quarter_plot
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import lag_plot
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.tools import diff
import holidays
import re
from pylab import rcParams

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

train = pd.read_csv("../input/demand-forecasting-kernels-only/train.csv", index_col= 'date', parse_dates= True)
test = pd.read_csv("../input/demand-forecasting-kernels-only/test.csv", index_col= 'date', parse_dates= True)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train['train_or_test'] = 'TRAIN'
test['train_or_test'] = 'TEST'

df = train.append(test, sort=False)
df.sort_values(by=['store','item','date'], axis=0, inplace=True)

#Creating lag features. These features just return the data from the lagged time period, i.e. 91, 98, 105 days ago.

df['lag_91'] = df.groupby(['store', 'item'])['sales'].shift(periods = 91)
df['lag_98'] = df.groupby(['store', 'item'])['sales'].shift(periods = 98)
df['lag_105'] = df.groupby(['store', 'item'])['sales'].shift(periods = 105)
df['lag_112'] = df.groupby(['store', 'item'])['sales'].shift(periods = 112)
df['lag_119'] = df.groupby(['store', 'item'])['sales'].shift(periods = 119)
df['lag_126'] = df.groupby(['store', 'item'])['sales'].shift(periods = 126)
df['lag_133'] = df.groupby(['store', 'item'])['sales'].shift(periods = 133)
df['lag_140'] = df.groupby(['store', 'item'])['sales'].shift(periods = 140)
df['lag_147'] = df.groupby(['store', 'item'])['sales'].shift(periods = 147)
df['lag_154'] = df.groupby(['store', 'item'])['sales'].shift(periods = 154)
df['lag_161'] = df.groupby(['store', 'item'])['sales'].shift(periods = 161)
df['lag_168'] = df.groupby(['store', 'item'])['sales'].shift(periods = 168)
df['lag_175'] = df.groupby(['store', 'item'])['sales'].shift(periods = 175)
df['lag_182'] = df.groupby(['store', 'item'])['sales'].shift(periods = 182)
df['lag_364'] = df.groupby(['store', 'item'])['sales'].shift(periods = 364)


#Creating rolling mean and median features

df['rolling_182_mean'] = df.groupby(['store', 'item'])['sales'].shift(periods = 91).rolling(window = 91).mean()
df['rolling_182_median'] = df.groupby(['store', 'item'])['sales'].shift(periods = 91).rolling(window = 91).median()
df['rolling_364_mean'] = df.groupby(['store', 'item'])['sales'].shift(periods = 91).rolling(window = 182).mean()
df['rolling_364_median'] = df.groupby(['store', 'item'])['sales'].shift(periods = 91).rolling(window = 182).median()


#Creating exponentially weighted average features

df['ewa_91_mean'] = df.groupby(['store', 'item'])['sales'].shift(periods = 91).ewm(span= 12).mean()
df['ewa_182_mean'] = df.groupby(['store', 'item'])['sales'].shift(periods = 182).ewm(span= 12).mean()
df['ewa_365_mean'] = df.groupby(['store', 'item'])['sales'].shift(periods = 364).ewm(span= 12).mean()

df['date'] = df.index
df['dom'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['dow'] = df['date'].dt.dayofweek
#f_df['weekofyear'] = f_df['date'].dt.weekofyear
df['is_month_start'] = (df['date'].dt.is_month_start).astype(int)
df['is_month_end'] = (df['date'].dt.is_month_end).astype(int)
df.drop('date', axis = 1, inplace = True)

df = pd.get_dummies(df, columns=['dom', 'month', 'dow'])

final_results = pd.Series()

train_final = df[df['train_or_test'] == "TRAIN"]
train_final.dropna(subset = ['lag_364'], inplace = True)

test_final = df[df['train_or_test'] == "TEST"]

for item in range(1,51):
    for store in range(1,11):
        current_item = train_final[(train_final['store']==store) & (train_final['item']==item)]
        test_item = test_final[(test_final['store']==store) & (test_final['item']==item)]
        current_item.index.freq = 'D'
        test_item.index.freq = 'D'
        
        start = len(current_item)
        end = len(current_item) + len(test_item) - 1
        
        model_final = SARIMAX(current_item['sales'], order=(0, 0, 0), seasonal_order=(2, 0, [1], 7), 
            exog= current_item[['rolling_182_mean', 'rolling_182_median','rolling_364_mean', 'rolling_364_median',
            'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9',
            'month_10', 'month_11', 'month_12', 'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6']],
            enforce_invertibility= False, enforce_stationarity=False).fit()
        
        predictions_final = model_final.predict(start, end, exog= test_item[['rolling_182_mean', 'rolling_182_median','rolling_364_mean',
       'rolling_364_median', 'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8',
       'month_9', 'month_10', 'month_11', 'month_12', 'dow_0', 'dow_1',
       'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6']], typ = 'levels')
        
        final_results = final_results.append(predictions_final)

csv = final_results.reset_index()
csv = csv.reset_index()
csv.rename(columns={'level_0':'id', 0:'sales'}, inplace=True)
csv.drop('index', axis = 1, inplace = True)

csv.to_csv('submission.csv', index = False )
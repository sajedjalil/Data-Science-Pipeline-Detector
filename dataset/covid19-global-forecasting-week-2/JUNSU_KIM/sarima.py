# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings(action='ignore')

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def RMSLE(pred,actual):
    return np.sqrt(np.mean(np.power((np.log(pred+1)-np.log(actual+1)),2)))
pd.set_option('mode.chained_assignment', None)
test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")
train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")
train['Province_State'].fillna('', inplace=True)
test['Province_State'].fillna('', inplace=True)
train['Date'] =  pd.to_datetime(train['Date'])
test['Date'] =  pd.to_datetime(test['Date'])
train = train.sort_values(['Country_Region','Province_State','Date'])
test = test.sort_values(['Country_Region','Province_State','Date'])

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA

feature_day = [1,20,50,100,200,500,1000]
def CreateInput(data):
    feature = []
    for day in feature_day:
        #Get information in train data
        data.loc[:,'Number day from ' + str(day) + ' case'] = 0
        if (train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].count() > 0):
            fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].max()        
        else:
            fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].min()       
        for i in range(0, len(data)):
            if (data['Date'].iloc[i] > fromday):
                day_denta = data['Date'].iloc[i] - fromday
                data['Number day from ' + str(day) + ' case'].iloc[i] = day_denta.days 
        feature = feature + ['Number day from ' + str(day) + ' case']
    
    return data[feature]

pred_data_all = pd.DataFrame()
for country in train['Country_Region'].unique():
#for country in ['Vietnam']:
    for province in train[(train['Country_Region'] == country)]['Province_State'].unique():
        df_train = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]
        df_test = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]
        X_train = CreateInput(df_train)
        y_train_confirmed = df_train['ConfirmedCases'].ravel()
        y_train_fatalities = df_train['Fatalities'].ravel()
        X_pred = CreateInput(df_test)
        
        # Only train above 50 cases
        for day in sorted(feature_day,reverse = True):
            feature_use = 'Number day from ' + str(day) + ' case'
            idx = X_train[X_train[feature_use] == 0].shape[0]     
            if (X_train[X_train[feature_use] > 0].shape[0] >= 20):
                break
                                           
        adjusted_X_train = X_train[idx:][feature_use].values.reshape(-1, 1)
        adjusted_y_train_confirmed = y_train_confirmed[idx:]
        adjusted_y_train_fatalities = y_train_fatalities[idx:] #.values.reshape(-1, 1)
        idx = X_pred[X_pred[feature_use] == 0].shape[0]    
        adjusted_X_pred = X_pred[idx:][feature_use].values.reshape(-1, 1)
        
        pred_data = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]
        max_train_date = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].max()
        min_test_date = pred_data['Date'].min()
        model = SARIMAX(adjusted_y_train_confirmed, order=(1,1,0), 
                        #seasonal_order=(1,1,0,12),
                        measurement_error=True).fit(disp=False)
        y_hat_confirmed = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])
        y_train_confirmed = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['ConfirmedCases'].values
        y_hat_confirmed = np.concatenate((y_train_confirmed,y_hat_confirmed), axis = 0)
               
        model = SARIMAX(adjusted_y_train_fatalities, order=(1,1,0), 
                        #seasonal_order=(1,1,0,12),
                        measurement_error=True).fit(disp=False)
        y_hat_fatalities = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])
        y_train_fatalities = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['Fatalities'].values
        y_hat_fatalities = np.concatenate((y_train_fatalities,y_hat_fatalities), axis = 0)
        
        
        pred_data['ConfirmedCases_hat'] =  y_hat_confirmed
        pred_data['Fatalities_hat'] = y_hat_fatalities
        pred_data_all = pred_data_all.append(pred_data)

df_val = pd.merge(pred_data_all,train[['Date','Country_Region','Province_State','ConfirmedCases','Fatalities']],on=['Date','Country_Region','Province_State'], how='left')
df_val.loc[df_val['Fatalities_hat'] < 0,'Fatalities_hat'] = 0
df_val.loc[df_val['ConfirmedCases_hat'] < 0,'ConfirmedCases_hat'] = 0
df_val_3 = df_val.copy()

submission = df_val[['ForecastId','ConfirmedCases_hat','Fatalities_hat']]
submission.columns = ['ForecastId','ConfirmedCases','Fatalities']
submission.to_csv('submission.csv', index=False)
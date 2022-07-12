import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import datetime
from datetime import timedelta
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from matplotlib import pyplot
import math
from math import sqrt

from time import time

import warnings 
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
data= pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
sub=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')

data['Province_State']=data['Province_State'].fillna('')
test['Province_State']=test['Province_State'].fillna('')

set(test['Province_State']).difference(set(data['Province_State']))

from datetime import datetime
datetime_str = '01/22/20 00:00:00'
datetime_object = datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')
data['days']=pd.to_datetime(data['Date']).sub(datetime_object)/np.timedelta64(1, 'D')
test['days']=pd.to_datetime(test['Date']).sub(datetime_object)/np.timedelta64(1, 'D')

data.loc[(data['Province_State']=='')&(data['Country_Region']=='India'),:].sort_values(by="Date")

countries=dict()
for cnt in data['Country_Region'].unique():
    countries[cnt]=data.loc[data['Country_Region']==cnt,'Province_State'].unique
    
data.head()

data['Date']=pd.to_datetime(data['Date'])
test['Date']=pd.to_datetime(test['Date'])

data['Date'].max()

data['Date'].min(), test['Date'].max(), test['Date'].min()

pd.DataFrame(data.loc[data['Country_Region']=='Afghanistan',['ConfirmedCases']]).reset_index(drop=True)

data.isna().sum(axis=0)

data['ConfirmedCases'][data['Country_Region']==''][51:]

data['ConfirmedCases'][data['Country_Region']=='Switzerland'].value_counts()

pd.DataFrame(data.loc[data['Country_Region']=='Switzerland',['ConfirmedCases']])

countries

datetime_str = '03/22/20 00:00:00'
datetime_object = datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')

def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

def evaluate_arima_model(X,forecast_days, arima_order):
    # prepare training dataset
    X=[x for x in X]
    train_size = int(len(X) * 0.9)
    train, test1 = X[0:train_size], X[train_size:]
    # make predictions
    history=train
    model = ARIMA(history, order=arima_order)
    model_fit = model.fit(disp=0)
    predictions = list()
    predictions =model_fit.forecast(steps=len(test1))[0]
    model = ARIMA(X, order=arima_order)
    model_fit = model.fit(disp=0)
    if np.isnan(model_fit.forecast(steps=forecast_days)[0]).sum()>0:
        return float('inf')
    error = rmsle(test1, predictions) 
    return error

def evaluate_models(dataset,forcast_days, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), (0,0,0)
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset,forcast_days, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                except:
                    continue 
        

    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return best_cfg, best_score

data.head()

test['ConfirmedCases']=0
test['Fatalities']=0

sliced_data=data.loc[(data['Province_State']=='')&(data['Country_Region']=='Switzerland'),:]

sliced_data.head()

country='Switzerland'
state=''
sliced_data=data.loc[(data['Province_State']==state)&(data['Country_Region']==country),:]
test_sliced=test.loc[(test['Province_State']==state)&(test['Country_Region']==country),:]
sliced_data=sliced_data.drop_duplicates()
sliced_data=sliced_data.reset_index(drop=True)
sliced_data=sliced_data.sort_values(by='Date')
if sliced_data.loc[sliced_data['ConfirmedCases']>0,:].shape[0]>0:
    sliced_data=sliced_data.loc[sliced_data['ConfirmedCases']>0,:]
sliced_data=sliced_data.reset_index(drop=True)
max_date_train=sliced_data['Date'].max()
max_date_test=test_sliced['Date'].max()
forcast_days=int((max_date_test-max_date_train)/np.timedelta64(1, 'D'))
history=sliced_data['ConfirmedCases'].to_list()

if len(history)==1:
    history.append(history[0])

best_cfg,best_score=evaluate_models(history,forcast_days,range(10),range(7),range(7))
preds=[]
model = ARIMA(history, order=best_cfg)
model_fit = model.fit(disp=0)
preds=model_fit.forecast(steps=forcast_days)[0]
preds=[round(p) if p>0 else 0 for p in preds]

dates=[max_date_train+timedelta(days=day+1) for day in range(forcast_days)]  
predictions=pd.DataFrame()
predictions['Date']=dates
predictions['ConfirmedCases']=preds
test_sliced=test_sliced.merge(sliced_data[['Date','ConfirmedCases']], on='Date',how='left')
test_sliced['ConfirmedCases']=test_sliced['ConfirmedCases_y']
del test_sliced['ConfirmedCases_y']
del test_sliced['ConfirmedCases_x']
test_sliced=test_sliced.merge(predictions, on='Date',how='left')
test_sliced['ConfirmedCases_x'][test_sliced['ConfirmedCases_x'].isna()]=test_sliced['ConfirmedCases_y'][test_sliced['ConfirmedCases_x'].isna()]
test_sliced['ConfirmedCases']=test_sliced['ConfirmedCases_x']
del test_sliced['ConfirmedCases_y']
del test_sliced['ConfirmedCases_x']

sliced_data_bck=sliced_data.copy()
if sliced_data.loc[sliced_data['Fatalities']>0,:].shape[0]>0:
    sliced_data=sliced_data.loc[sliced_data['Fatalities']>0,:]
sliced_data=sliced_data.reset_index(drop=True)
max_date_train=sliced_data['Date'].max()
max_date_test=test_sliced['Date'].max()
forcast_days=int((max_date_test-max_date_train)/np.timedelta64(1, 'D'))
history=sliced_data['Fatalities'].to_list()

if len(history)==1:
    history.append(history[0])

best_cfg,best_score=evaluate_models(history,forcast_days,range(5),range(5),range(5))
preds=[]
model=None
model = ARIMA(history, order=best_cfg)
model_fit = model.fit(disp=0)
preds=model_fit.forecast(steps=forcast_days)[0]
preds=[round(p) if p>0 else 0 for p in preds]

dates=[max_date_train+timedelta(days=day+1) for day in range(forcast_days)]  
predictions_f=pd.DataFrame()
predictions_f['Date']=dates
predictions_f['Fatalities']=preds
test_sliced=test_sliced.merge(sliced_data_bck[['Date','Fatalities']], on='Date',how='left')
test_sliced['Fatalities']=test_sliced['Fatalities_y']
del test_sliced['Fatalities_y']
del test_sliced['Fatalities_x']
test_sliced=test_sliced.merge(predictions_f, on='Date',how='left')
test_sliced['Fatalities_x'][test_sliced['Fatalities_x'].isna()]=test_sliced['Fatalities_y'][test_sliced['Fatalities_x'].isna()]
test_sliced['Fatalities']=test_sliced['Fatalities_x']
del test_sliced['Fatalities_y']
del test_sliced['Fatalities_x']
test=test.merge(test_sliced,on='ForecastId',how='left')
test['ConfirmedCases_x'][test['ConfirmedCases_y'].notna()]=test['ConfirmedCases_y'][test['ConfirmedCases_y'].notna()]
test['Fatalities_x'][test['Fatalities_y'].notna()]=test['Fatalities_y'][test['Fatalities_y'].notna()]
new_cols=[]
for col in test.columns:
    if col[-2:]=='_y':
        del test[col]
    elif col[-2:]=='_x':
        new_cols.append(col[:-2])
    else:
        new_cols.append(col)
test.columns=new_cols
test.loc[(test['Province_State']==state)&(test['Country_Region']==country),:].head()

test.loc[(test['Province_State']==state)&(test['Country_Region']==country),['Country_Region','Date','ConfirmedCases','Fatalities']]

test['ConfirmedCases']=0
test['Fatalities']=0

for country in countries:
    for state in countries[country]:
        print('Country : '+country,'State : '+state)
        sliced_data=data.loc[(data['Province_State']==state)&(data['Country_Region']==country),:]
        test_sliced=test.loc[(test['Province_State']==state)&(test['Country_Region']==country),:]
        sliced_data=sliced_data.drop_duplicates()
        sliced_data=sliced_data.reset_index(drop=True)
        sliced_data=sliced_data.sort_values(by='Date')
        if sliced_data.loc[sliced_data['ConfirmedCases']>0,:].shape[0]>0:
            sliced_data=sliced_data.loc[sliced_data['ConfirmedCases']>0,:]
        sliced_data=sliced_data.reset_index(drop=True)
        max_date_train=sliced_data['Date'].max()
        max_date_test=test_sliced['Date'].max()
        forcast_days=int((max_date_test-max_date_train)/np.timedelta64(1, 'D'))
        history=sliced_data['ConfirmedCases'].to_list()
        
        if len(history)==1:
            history.append(history[0])
            
        best_cfg,best_score=evaluate_models(history,forcast_days,range(5),range(5),range(5))
        preds=[]
        model = ARIMA(history, order=best_cfg)
        model_fit = model.fit(disp=0)
        preds=model_fit.forecast(steps=forcast_days)[0]
        preds=[round(p) if p>0 else 0 for p in preds]
            
        dates=[max_date_train+timedelta(days=day+1) for day in range(forcast_days)]  
        predictions=pd.DataFrame()
        predictions['Date']=dates
        predictions['ConfirmedCases']=preds
        test_sliced=test_sliced.merge(sliced_data[['Date','ConfirmedCases']], on='Date',how='left')
        test_sliced['ConfirmedCases']=test_sliced['ConfirmedCases_y']
        del test_sliced['ConfirmedCases_y']
        del test_sliced['ConfirmedCases_x']
        test_sliced=test_sliced.merge(predictions, on='Date',how='left')
        test_sliced['ConfirmedCases_x'][test_sliced['ConfirmedCases_x'].isna()]=test_sliced['ConfirmedCases_y'][test_sliced['ConfirmedCases_x'].isna()]
        test_sliced['ConfirmedCases']=test_sliced['ConfirmedCases_x']
        del test_sliced['ConfirmedCases_y']
        del test_sliced['ConfirmedCases_x']
        
        sliced_data_bck=sliced_data.copy()
        if sliced_data.loc[sliced_data['Fatalities']>0,:].shape[0]>0:
            sliced_data=sliced_data.loc[sliced_data['Fatalities']>0,:]
        sliced_data=sliced_data.reset_index(drop=True)
        max_date_train=sliced_data['Date'].max()
        max_date_test=test_sliced['Date'].max()
        forcast_days=int((max_date_test-max_date_train)/np.timedelta64(1, 'D'))
        history=sliced_data['Fatalities'].to_list()
        
        if len(history)==1:
            history.append(history[0])
        
        best_cfg,best_score=evaluate_models(history,forcast_days,range(5),range(5),range(5))
        preds=[]
        model=None
        model = ARIMA(history, order=best_cfg)
        model_fit = model.fit(disp=0)
        preds=model_fit.forecast(steps=forcast_days)[0]
        preds=[round(p) if p>0 else 0 for p in preds]
            
        dates=[max_date_train+timedelta(days=day+1) for day in range(forcast_days)]  
        predictions_f=pd.DataFrame()
        predictions_f['Date']=dates
        predictions_f['Fatalities']=preds
        test_sliced=test_sliced.merge(sliced_data_bck[['Date','Fatalities']], on='Date',how='left')
        test_sliced['Fatalities']=test_sliced['Fatalities_y']
        del test_sliced['Fatalities_y']
        del test_sliced['Fatalities_x']
        test_sliced=test_sliced.merge(predictions_f, on='Date',how='left')
        test_sliced['Fatalities_x'][test_sliced['Fatalities_x'].isna()]=test_sliced['Fatalities_y'][test_sliced['Fatalities_x'].isna()]
        test_sliced['Fatalities']=test_sliced['Fatalities_x']
        del test_sliced['Fatalities_y']
        del test_sliced['Fatalities_x']
        test=test.merge(test_sliced,on='ForecastId',how='left')
        test['ConfirmedCases_x'][test['ConfirmedCases_y'].notna()]=test['ConfirmedCases_y'][test['ConfirmedCases_y'].notna()]
        test['Fatalities_x'][test['Fatalities_y'].notna()]=test['Fatalities_y'][test['Fatalities_y'].notna()]
        new_cols=[]
        for col in test.columns:
            if col[-2:]=='_y':
                del test[col]
            elif col[-2:]=='_x':
                new_cols.append(col[:-2])
            else:
                new_cols.append(col)
        test.columns=new_cols
        print(test.loc[(test['Province_State']==state)&(test['Country_Region']==country),:].head())
        
        
output=pd.DataFrame()
output['ForecastId']=test['ForecastId'].astype(int)
output['ConfirmedCases']=test['ConfirmedCases'].astype(int)
output['Fatalities']=test['Fatalities'].astype(int)
output.to_csv('submission.csv',index=False)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fbprophet import Prophet

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# Read Data

cal_df = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv")
cal_test= pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv")

## prepare data for FB-Prophet 
## Confirm data
cal_dat_fb = pd.DataFrame({'ds':cal_df['Date'],'y':cal_df['ConfirmedCases']})
cal_dat_fb['ds'] = cal_dat_fb['ds'].apply(pd.to_datetime)

## Fatalities
cal_dat_fat = pd.DataFrame({'ds':cal_df['Date'],'y':cal_df['Fatalities']})
cal_dat_fat['ds'] = cal_dat_fat['ds'].apply(pd.to_datetime)
    
#future data
future = pd.DataFrame({'ds':cal_test['Date']})

## FB prophet code

### Confirmed cases
m1 = Prophet(mcmc_samples=1000,interval_width=0.01,yearly_seasonality=0.4)
m2 = Prophet(mcmc_samples=1000,interval_width=0.01,yearly_seasonality=0.4)
Confirm_forecast = m1.fit(cal_dat_fb[48:]).predict(future)
Fatality_forecast = m2.fit(cal_dat_fat[48:]).predict(future)

#cf= pd.DataFrame(columns=['ConfirmedCases'])
#ff=pd.DataFrame(columns=['Fatalities'])
cf = Confirm_forecast[['yhat']].apply(np.ceil)
ff=Fatality_forecast[['yhat']].apply(np.ceil)
cf= cf.rename(columns={"yhat": "ConfirmedCases"})
ff= ff.rename(columns={"yhat": "Fatalities"})

print(Fatality_forecast[['yhat']])

## submission code 
dlist = [cal_test['ForecastId'],cf['ConfirmedCases'],ff['Fatalities']]
#names = ['ForecastId','ConfirmedCases','Fatalities']
submission = pd.concat(dlist,axis=1)

print(submission)
submission.to_csv('submission.csv',index=False)
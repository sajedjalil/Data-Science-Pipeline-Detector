import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import requests
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kurtosis
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os
from sklearn.manifold import TSNE
import seaborn as sns
from datetime import datetime, timedelta
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.
# Data Source
# https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide
pd.options.display.max_columns = 50
pd.options.display.max_rows = 100
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv",encoding = "ISO-8859-1")
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv",encoding = "ISO-8859-1")
submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv",encoding = "ISO-8859-1")

test['Date'] = pd.to_datetime(test.Date, infer_datetime_format=True)
train['Date'] = pd.to_datetime(train.Date, infer_datetime_format=True)
train['doy'] = train['Date'].dt.dayofyear
train['dow'] = train.Date.dt.dayofweek
train['moy'] = train.Date.dt.month
train['woy'] = train.Date.dt.weekofyear
train['is_weekend'] = np.where(train['Date'].dt.weekday_name.isin(['Sunday','Saturday']),1,0)
train['Province_State'] = train['Province_State'].fillna('')
train['Country_Region_Province_State'] = train[['Country_Region','Province_State']].apply(lambda x: x[0]+"_"+x[1],axis = 1)

doa = train[train['ConfirmedCases']>0].groupby('Country_Region_Province_State')['Date'].min().reset_index()
doa.columns = ['Country_Region_Province_State','doa']

train = train.merge(doa,how = 'left',on='Country_Region_Province_State')
train['day_since_doa'] = (train.Date - train.doa).dt.days
#train = train[train['day_since_doa']>=0]
train['log_Fatalities'] = train['Fatalities'].apply(lambda x: np.log(x+1))
train['daily_changef'] = (train.groupby('Country_Region_Province_State')['log_Fatalities'].apply(pd.Series.pct_change)).replace(np.NaN,0).replace(np.inf,1)
train['actual_casesf'] = train.groupby(['Country_Region_Province_State'])['log_Fatalities'].diff().fillna(0)

train['log_ConfirmedCases'] = train['ConfirmedCases'].apply(lambda x: np.log(x+1))
train['daily_changeC'] = (train.groupby('Country_Region_Province_State')['log_ConfirmedCases'].apply(pd.Series.pct_change,1)).replace(np.NaN,0).replace(np.inf,1)
train['actual_casesC'] = train.groupby(['Country_Region_Province_State'])['log_ConfirmedCases'].diff().fillna(0)

train['ratio'] = (train['log_Fatalities']/train['log_ConfirmedCases']).replace(np.nan,0)

train['log_fatalities_to_doa'] = (train['log_Fatalities']/train['doy'].apply(lambda x: x if (x >0) else 0)).replace(np.NaN,0).replace(np.inf,1)
train['log_ConfirmedCases_to_doa'] = (train['log_ConfirmedCases']/train['doy'].apply(lambda x: x if (x >0) else 0)).replace(np.NaN,0).replace(np.inf,1)


common_date = list(set(test.Date.unique()) & set(train.Date.unique()))
total_country_region = train.Country_Region_Province_State.unique().tolist() 
#dft = pd.pivot_table(train,values='Fatalities',fill_value=0,index = 'Country_Region_Province_State',columns='Date',aggfunc='sum')
stop_date = test.Date.min()  # " < than this date"  # Timestamp('2020-04-02 00:00:00')
start_date = stop_date - timedelta(days=40) #" >= this date "

trn_training_strt_date = start_date
trn_training_end_date = start_date + timedelta(days=9)
val_training_strt_date = start_date + timedelta(days=10)
val_training_end_date = start_date + timedelta(days=19)

tst_feature_strt_date = stop_date - timedelta(days=43)
tst_feature_end_date = stop_date
tst_feature_strt_date = stop_date - timedelta(days=43)

train = train[train['Date']<stop_date]
train = train[train['Date']>=start_date]



test['Date'] = pd.to_datetime(test.Date, infer_datetime_format=True)
test['doy'] = test['Date'].dt.dayofyear
test['dow'] = test.Date.dt.dayofweek
test['moy'] = test.Date.dt.month
test['woy'] = test.Date.dt.weekofyear
test['is_weekend'] = np.where(test['Date'].dt.weekday_name.isin(['Sunday','Saturday']),1,0)

test['Province_State'] = test['Province_State'].fillna('')
test['Country_Region_Province_State'] = test[['Country_Region','Province_State']].apply(lambda x: x[0]+"_"+x[1],axis = 1)

test = test.merge(doa,how = 'left',on='Country_Region_Province_State')
test['day_since_doa'] = (test.Date - test.doa).dt.days

test['map'] = test[['Country_Region_Province_State','doy']].apply(lambda x: x[0]+"_"+str(x[1]),axis = 1)


FEAT =  train[(train['Date']>=tst_feature_strt_date) & (train['Date']<=tst_feature_end_date)]

FEAT.doy.unique().tolist()
def get_poly(df,feat_val):
    rng =FEAT.doy.unique().tolist()
    t = pd.pivot_table(df,values=feat_val,fill_value=0,index = 'Country_Region_Province_State',columns='doy',aggfunc='sum')
    ss1 = t.apply(lambda x: np.polyfit(rng, x, 2),axis = 1).reset_index()
    ss2 =  t.apply(lambda x: x.tolist(),axis = 1).reset_index()
    
    s = ss1.merge(ss2,on = ['Country_Region_Province_State'])
    #s.apply(lambda x: x[1])
    ss = s.apply(lambda x: np.poly1d(x[1])(np.arange(93,136)),axis = 1)
    
    dt = pd.DataFrame(list(map(np.ravel, [i for i in ss])))
    dt['Country_Region_Province_State'] = t.index.tolist()
    dt.columns = [int(name)+93 if idx<dt.shape[1]-1 else name for idx,name in enumerate(dt.columns.tolist())]
    return dt
  
pred_C  = get_poly(FEAT,'log_ConfirmedCases')
pred_f  = get_poly(FEAT,'log_Fatalities')

pred_f = pred_f.melt(id_vars =['Country_Region_Province_State'])
pred_C = pred_C.melt(id_vars =['Country_Region_Province_State'])

pred_f.columns = ['Country_Region_Province_State', 'Date', 'pred_Fatalities']
pred_C.columns = ['Country_Region_Province_State', 'Date', 'pred_ConfirmedCases']

pred_f = pred_f.sort_values(['Country_Region_Province_State','Date'])
pred_C = pred_C.sort_values(['Country_Region_Province_State','Date'])

pred = pred_C.merge(pred_f, on = 'Country_Region_Province_State')

test = test.merge(pred,left_on = ['Country_Region_Province_State','doy'],right_on = ['Country_Region_Province_State','Date_x'])

submission['Fatalities'] = test['pred_Fatalities'].apply(lambda x: int(np.exp(x)-1))
submission['ConfirmedCases'] = test['pred_ConfirmedCases'].apply(lambda x: int(np.exp(x)-1))

submission.to_csv("submission.csv",index = False)

print("Stop")
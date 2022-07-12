import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.preprocessing import LabelBinarizer,LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression,SGDClassifier,LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
import keras
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense,LSTM
import tensorflow as tf

train_df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
test_df = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
submission = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

#%%
ans_df = train_df[(train_df['Date']>'2020-04-01')][['ConfirmedCases','Fatalities']]
val_df = test_df[(test_df['Date']<='2020-04-09')]
#train_df = train_df[train_df['Date']<='2020-04-01']

train_df['Province_State'].fillna("",inplace = True)
test_df['Province_State'].fillna("",inplace = True)
val_df['Province_State'].fillna("",inplace = True)

train_df['Country_Region'] = train_df['Country_Region'] + ' ' + train_df['Province_State']
test_df['Country_Region'] = test_df['Country_Region'] + ' ' + test_df['Province_State']
val_df['Country_Region'] = val_df['Country_Region'] + ' ' + val_df['Province_State']
del train_df['Province_State']
del test_df['Province_State']
del val_df['Province_State']

train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])
val_df['Date'] = pd.to_datetime(val_df['Date'])

train_df['Month'] = train_df['Date'].dt.month
train_df['Day'] = train_df['Date'].dt.day
train_df['Weekday'] = train_df['Date'].dt.weekday

test_df['Month'] = test_df['Date'].dt.month
test_df['Day'] = test_df['Date'].dt.day
test_df['Weekday'] = test_df['Date'].dt.weekday

val_df['Month'] = val_df['Date'].dt.month
val_df['Day'] = val_df['Date'].dt.day
val_df['Weekday'] = val_df['Date'].dt.weekday

del train_df['Date']
del train_df['Id']

del test_df['Date']
del test_df['ForecastId']

del val_df['Date']
del val_df['ForecastId']

lb = LabelEncoder()
train_df['Country_Region'] = lb.fit_transform(train_df['Country_Region'])
test_df['Country_Region'] = lb.transform(test_df['Country_Region'])
val_df['Country_Region'] = lb.transform(val_df['Country_Region'])

#%%任意のフィッティング_国ごと
try:
    del train_df['Date']
    del test_df['Date']
    del val_df['Date']
except:
    pass

cases_pred = []
fatalities_pred = []

cases_pred_val = []
fatalities_pred_val = []


from scipy.optimize import curve_fit
def one_function(x, A, B):
    return  A*x + B

def two_function(x, A, B, C):
    return  A*x*x + B*x + C

#0.4317045980701612
def function(x, A, B, C, D):
    return D + (A - D)/(1 + (x/C)**B)

for i in train_df['Country_Region'].unique():
    print(i)
    #ConfirmedCases
    df_train_cases = train_df[train_df['Country_Region']==i]
    df_test_cases = test_df[test_df['Country_Region']==i]
    df_train_cases = df_train_cases[(df_train_cases['ConfirmedCases']!=0)|((df_train_cases['Month']>=4)&(df_train_cases['Day']>=2))]
    '''
    if df_train_cases.shape[0]==0:
        df_train_cases = train_df[train_df['Country_Region']==i]
    '''
    x = range(0,len(df_train_cases))
    x_val = range(len(df_train_cases),len(df_train_cases)+8)
    x_test = range(len(df_train_cases)-12,len(df_train_cases)+len(df_test_cases)-12)
    
    y = np.array(df_train_cases['ConfirmedCases'])
    try:
        param, cov = curve_fit(function, x, y, maxfev=10000)
        cases_pred.append(function(x_test,*param))
        cases_pred_val.append(function(x_val,*param))
    except:
        try:
            param, cov = curve_fit(two_function, x, y, maxfev=10000)
            cases_pred.append(two_function(x_test,*param))
            cases_pred_val.append(two_function(x_val,*param))    
        except:
            param, cov = curve_fit(one_function, x, y)
            cases_pred.append(one_function(x_test,*param))
            cases_pred_val.append(one_function(x_val,*param))  
    #Fatalities
    y = np.array(df_train_cases['Fatalities'])
    try:
        param, cov = curve_fit(function, x, y, maxfev=10000)
        fatalities_pred.append(function(x_test,*param))
        fatalities_pred_val.append(function(x_val,*param))
    except:
        try:
            param, cov = curve_fit(two_function, x, y, maxfev=10000)
            fatalities_pred.append(two_function(x_test,*param))
            fatalities_pred_val.append(two_function(x_val,*param))
        except:
            param, cov = curve_fit(one_function, x, y)
            fatalities_pred.append(one_function(x_test,*param))
            fatalities_pred_val.append(one_function(x_val,*param))
            
cases_pred = np.around(cases_pred,decimals = 0)
cases_pred = cases_pred.reshape(-1,1)

cases_pred_val = np.around(cases_pred_val,decimals = 0)
cases_pred_val = cases_pred_val.reshape(-1,1)

fatalities_pred = np.around(fatalities_pred,decimals = 0)
fatalities_pred = fatalities_pred.reshape(-1,1)

fatalities_pred_val = np.around(fatalities_pred_val,decimals = 0)
fatalities_pred_val = fatalities_pred_val.reshape(-1,1)

cases_pred[cases_pred<0] = 0
fatalities_pred[fatalities_pred<0] = 0
cases_pred_val[cases_pred_val<0] = 0
fatalities_pred_val[fatalities_pred_val<0] = 0

submission['ConfirmedCases'] = cases_pred
submission['Fatalities'] = fatalities_pred

submission.to_csv("submission.csv" , index = False)

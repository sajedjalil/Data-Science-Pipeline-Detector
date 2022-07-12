# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Any results you write to the current directory are saved as output.

print("Read in libraries")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn import preprocessing

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from random import random

import datetime
import lightgbm as lgb
print("Print Directories")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
sub = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")
print("read in files")
test.head()

#At this part we append our testing data to the training data at the date where this is no overlap. Today the train set went up to March 28th.
train = train.append(test[test['Date']>'2020-03-31'])

#fix dates
import datetime 
train['Date'] = pd.to_datetime(train['Date'], format='%Y-%m-%d')
train['day_dist'] = train['Date']-train['Date'].min()
train['day_dist'] = train['day_dist'].dt.days
print(train['Date'].max())
#print(val['Date'].max())
print(test['Date'].min())
print(test['Date'].max())
#print(test['Date'].max()-test['Date'].min())
cat_cols = train.dtypes[train.dtypes=='object'].keys()
cat_cols

#fix na
for cat_col in cat_cols:
    train[cat_col].fillna('no_value', inplace = True)

#make a place variable 
train['place'] = train['Province_State']+'_'+train['Country_Region']
#vcheck = train[(train['Date']>='2020-03-12')]
#get the cat columns
cat_cols = train.dtypes[train.dtypes=='object'].keys()
cat_cols

#label the columns
for cat_col in ['place']:
    #train[cat_col].fillna('no_value', inplace = True) #train[cat_col].value_counts().idxmax()
    le = preprocessing.LabelEncoder()
    le.fit(train[cat_col])
    train[cat_col]=le.transform(train[cat_col])
    
#check train keys 
train.keys()

#set columns were going to drop during our training stage
drop_cols = ['Id', 'ConfirmedCases','Date', 'ForecastId','Fatalities','day_dist', 'Province_State', 'Country_Region']
#At this point you want to set your validation set to be the section of time that is in the overlap period with train

#val = train[(train['Id']).isnull()==True]
#train = train[(train['Id']).isnull()==False]
val = train[(train['Date']>='2020-03-19')&(train['Id'].isnull()==False)]
#test = train[(train['Date']>='2020-03-12')&(train['Id'].isnull()==True)]
#train = train[(train['Date']<'2020-03-22')&(train['Id'].isnull()==False)]
#val = train
val
y_ft = train["Fatalities"]
y_val_ft = val["Fatalities"]

y_cc = train["ConfirmedCases"]
y_val_cc = val["ConfirmedCases"]

#train.drop(drop_cols, axis=1, inplace=True)
#test.drop(drop_cols, axis=1, inplace=True)
#val.drop(drop_cols, axis=1, inplace=True)
y_val_ft
#define scoring functions
def rmsle (y_true, y_pred):
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))

def mape (y_true, y_pred):
    return np.mean(np.abs(y_pred -y_true)*100/(y_true+1))
#set params for lgbt
params = {
    "objective": "regression",
    "boosting": 'gbdt', #"gbdt",
    "num_leaves": 1280,
    "learning_rate": 0.05,
    "feature_fraction": 0.9, # 0.9,
    "reg_lambda": 2,
    "metric": "rmse",
    'min_data_in_leaf':20
}
#get dates for iterating over 
dates = test['Date'].unique()
dates

#Another tricky part, set this to be the same date as the stacked test data set from the beginning.
#subset them for the relevant dates
dates = dates[dates>'2020-03-31']
dates
len(dates)
train.isnull().sum()
i = 0
fold_n = 0
for date in dates:

    fold_n = fold_n +1 
    i = i+1
    if i==1:
        nrounds = 200
    else:
        nrounds = 100
    print(i)
    print(nrounds)
    train['shift_1_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i)
    train['shift_2_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+1)
    train['shift_3_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+2)
    train['shift_4_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+3)
    train['shift_5_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+4)
        
    val2 = train[train['Date']==date]
    train2 = train[(train['Date']<date)]
    y_cc = train2["ConfirmedCases"]
    #y_val_cc = val2["ConfirmedCases"]
    
    train2.drop(drop_cols, axis=1, inplace=True)
    val2.drop(drop_cols, axis=1, inplace=True)
    
    #np.log1p(y)
    #feature_importances = pd.DataFrame()
    #feature_importances['feature'] = train.keys()
    
    #score = 0       
    dtrain = lgb.Dataset(train2, label=y_cc)
    dvalid = lgb.Dataset(val2, label=y_val_cc)

    model = lgb.train(params, dtrain, nrounds, 
                            #valid_sets = [dtrain, dvalid],
                            categorical_feature = ['place'], #'Province/State', 'Country/Region'
                            verbose_eval=False)#, early_stopping_rounds=50)

    y_pred = model.predict(val2,num_iteration=nrounds)  #model.best_iteration
    #y_pred = np.expm1( y_pred)
    #vcheck.loc[vcheck['Date']==date,'cc_predict'] = y_pred
    test.loc[test['Date']==date,'ConfirmedCases'] = y_pred
    train.loc[train['Date']==date,'ConfirmedCases'] = y_pred
    #y_oof[valid_index] = y_pred

    #rmsle_score = rmsle(y_val_cc, y_pred)
    #mape_score = mape(y_val_cc, y_pred)
    #score += rmsle_score
    #print (f'fold: {date}, rmsle: {rmsle_score:.5f}' )
    #print (f'fold: {date}, mape: {mape_score:.5f}' )
#y_pred = model.predict(val2,num_iteration=nrounds) 
test[test['Country_Region']=='Italy']
y_pred.mean()
i = 0
fold_n = 0
for date in dates:

    fold_n = fold_n +1 
    i = i+1
    if i==1:
        nrounds = 200
    else:
        nrounds = 100
    print(i)
    print(nrounds)
    
    train['shift_1_cc'] = train.groupby(['place'])['Fatalities'].shift(i)
    train['shift_2_cc'] = train.groupby(['place'])['Fatalities'].shift(i+1)
    train['shift_3_cc'] = train.groupby(['place'])['Fatalities'].shift(i+2)
    train['shift_4_cc'] = train.groupby(['place'])['Fatalities'].shift(i+3)
    train['shift_5_cc'] = train.groupby(['place'])['Fatalities'].shift(i+4)
        
    val2 = train[train['Date']==date]
    train2 = train[(train['Date']<date)]
    y_ft = train2["Fatalities"]
    #y_val_cc = val2["ConfirmedCases"]
    
    train2.drop(drop_cols, axis=1, inplace=True)
    val2.drop(drop_cols, axis=1, inplace=True)
    
    #np.log1p(y)
    #feature_importances = pd.DataFrame()
    #feature_importances['feature'] = train.keys()
    
    #score = 0       
    dtrain = lgb.Dataset(train2, label=y_ft)
    dvalid = lgb.Dataset(val2, label=y_val_ft)

    model = lgb.train(params, dtrain, nrounds, 
                            #valid_sets = [dtrain, dvalid],
                            categorical_feature = ['place'], #'Province/State', 'Country/Region'
                            verbose_eval=False)#, early_stopping_rounds=50)

    y_pred = model.predict(val2,num_iteration=nrounds)  #model.best_iteration
    #y_pred = np.expm1( y_pred)
    #vcheck.loc[vcheck['Date']==date,'cc_predict'] = y_pred
    test.loc[test['Date']==date,'Fatalities'] = y_pred
    train.loc[train['Date']==date,'Fatalities'] = y_pred
    #y_oof[valid_index] = y_pred

    #rmsle_score = rmsle(y_val_cc, y_pred)
    #mape_score = mape(y_val_cc, y_pred)
    #score += rmsle_score
    #print (f'fold: {date}, rmsle: {rmsle_score:.5f}' )
    #print (f'fold: {date}, mape: {mape_score:.5f}' )
test[test['Country_Region']=='Italy']
print(len(test))
train_sub = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")
#train_sub.loc[(train_sub['Date']=='2020-03-24')&(train_sub['Country_Region']=='France')&(train_sub['Province_State']=='France'),'ConfirmedCases'] = 22654
#train_sub.loc[(train_sub['Date']=='2020-03-24')&(train_sub['Country_Region']=='France')&(train_sub['Province_State']=='France'),'Fatalities'] = 1000
test = pd.merge(test,train_sub[['Province_State','Country_Region','Date','ConfirmedCases','Fatalities']], on=['Province_State','Country_Region','Date'], how='left')
print(len(test))
test.head()
test.loc[test['ConfirmedCases_x'].isnull()==True]
test.loc[test['ConfirmedCases_x'].isnull()==True, 'ConfirmedCases_x'] = test.loc[test['ConfirmedCases_x'].isnull()==True, 'ConfirmedCases_y']
test.head()
test.loc[test['Fatalities_x'].isnull()==True, 'Fatalities_x'] = test.loc[test['Fatalities_x'].isnull()==True, 'Fatalities_y']
dates
#last_amount = test.loc[(test['Country_Region']=='Italy')&(test['Date']=='2020-03-24'),'ConfirmedCases_x']
#last_fat = test.loc[(test['Country_Region']=='Italy')&(test['Date']=='2020-03-24'),'Fatalities_x']
#last_fat.values[0]
#dates
#len(dates)
#30/29
#i = 0
#k = 35
#for date in dates:
#    k = k-1
#    i = i + 1
#    test.loc[(test['Country_Region']=='Italy')&(test['Date']==date),'ConfirmedCases_x'] =  last_amount.values[0]+i*(5000-(100*i))
#    test.loc[(test['Country_Region']=='Italy')&(test['Date']==date),'Fatalities_x'] =  last_fat.values[0]+i*(800-(10*i))
#test.loc[(test['Country_Region']=='Italy')] #&(test['Date']==date),'ConfirmedCases_x' 
sub = test[['ForecastId', 'ConfirmedCases_x','Fatalities_x']]
sub.columns = ['ForecastId', 'ConfirmedCases', 'Fatalities']
sub.head()
sub.loc[sub['ConfirmedCases']<0, 'ConfirmedCases'] = 0
sub.loc[sub['Fatalities']<0, 'Fatalities'] = 0
sub['Fatalities'].describe()
sub.dtypes
#rename submission columns 
#sub = sub.rename(columns={'ForecastId': 'newName1', 'ConfirmedCases': 'Fatalities'})
sub.to_csv('submission.csv',index=False)

#make complete test file 
test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
complete_test= pd.merge(test, sub, how="left", on="ForecastId")
complete_test.to_csv('complete_test.csv',index=False)
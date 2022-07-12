#Jeremy Brouillet
#A notebook to calculate COVID-19 spread across the world.
#A logistic function is used to fit confirmed cases and fatalities for each state over the time period.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import scipy
from sklearn import linear_model
from scipy.optimize import curve_fit
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf
import math

# Import data files.
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')

#Rename columns
train.rename(columns={'Country_Region':'Country'}, inplace=True)
test.rename(columns={'Country_Region':'Country'}, inplace=True)

train.rename(columns={'Province_State':'State'}, inplace=True)
test.rename(columns={'Province_State':'State'}, inplace=True)

train['Date'] = pd.to_datetime(train['Date'], infer_datetime_format=True)
test['Date'] = pd.to_datetime(test['Date'], infer_datetime_format=True)

#Days were counted from the first global incidence of COVID-19
FIRST_COVID_DATE = pd.to_datetime('2019-11-17', infer_datetime_format=True) #Source : https://www.theguardian.com/world/2020/mar/13/first-covid-19-case-happened-in-november-china-government-records-show-report
train['DaysNumber']=(train['Date'].map(lambda x: (x - FIRST_COVID_DATE).days))
test['DaysNumber']=(test['Date'].map(lambda x: (x - FIRST_COVID_DATE).days))

#Overall structure:
#for country, state
    #fit each curve
    #make prediction
#write output

# Logistic function
def logisticR(x, a, b, c):
     return a / (1 + np.exp(-(x-b)*c))

#Initialize submission Calc
train_calc = pd.DataFrame(columns = ['ForecastId', 'ConfirmedCases','Fatalities'])
test_calc = pd.DataFrame(columns = ['ForecastId', 'ConfirmedCases','Fatalities'])
optimalparamsAll = pd.DataFrame(columns = ['Country','State','ParamsFatalities','ParamsCases','CovFatalities','CovCases'])

# Find trends by state
for country in set(train['Country']):   
    train_country = (train.loc[train['Country']==country])
    test_country = (test.loc[test['Country']==country])
    for state in set(train_country["State"]):
        if type(state)==float:
            train_state = train_country.loc[train_country['State'].map(lambda x: type(x)==float)]
            test_state =  test_country.loc[test_country['State'].map(lambda x: type(x)==float)]
            state_name = ''
        else:
            train_state = (train_country.loc[train_country['State']==state])
            test_state = (test_country.loc[test_country['State']==state])
            state_name = state

# Seperate out train/test data for each state
        xtrain = train_state["DaysNumber"]
        ytrainfatalities = train_state["Fatalities"]
        ytrainconfirmedcases = train_state["ConfirmedCases"]
        xtest = test_state["DaysNumber"]   

# Fit data to logistic to a logistic function
# Reasonable initial conditions and bounds were chosen. 
        optimalparamsCC, pcovarianceCC = curve_fit(logisticR, xtrain, ytrainconfirmedcases, maxfev=5000, p0=[0,130,.2], bounds = ([0,0,.08],[2000000,175,1]))
        maxDeaths = optimalparamsCC[0]

#Offsets: Fatalities starting point based on the cases
#The number of fatalities will be upper bounded by the number of confirmed cases
        startingPoint = [optimalparamsCC[0]*0.03,optimalparamsCC[1]+8,optimalparamsCC[2]]
        optimalparamsfatalities, pcovariancefatalities = curve_fit(logisticR, xtrain, ytrainfatalities, maxfev=5000, p0=startingPoint, bounds = ([0,0,.08],[maxDeaths,183,1]))
        
# Calculate out predictions
        ytrainestimatedfatalities=logisticR(xtrain, *optimalparamsfatalities)
        ytrainestimatedCC=logisticR(xtrain, *optimalparamsCC)
        ytestestimatedfatalities=logisticR(xtest, *optimalparamsfatalities)
        ytestestimatedCC=logisticR(xtest, *optimalparamsCC)
        
        train_calc_add = pd.DataFrame(columns = ['Id', 'ConfirmedCases','Fatalities'])
        train_calc_add["Fatalities"] = ytrainestimatedfatalities
        train_calc_add["ConfirmedCases"] = ytrainestimatedCC
        train_calc_add["Id"] = train_state["Id"]
        train_calc=train_calc.append(train_calc_add)
                
        test_calc_add = pd.DataFrame(columns = ['ForecastId', 'ConfirmedCases','Fatalities'])
        test_calc_add["Fatalities"] = ytestestimatedfatalities
        test_calc_add["ConfirmedCases"] = ytestestimatedCC
        test_calc_add["ForecastId"] = test_state["ForecastId"]
        test_calc=test_calc.append(test_calc_add)

        optimalparams_add = pd.DataFrame({'Country':[country],'State':[state],'ParamsFatalities':[optimalparamsfatalities],'ParamsCases':[optimalparamsCC],
                              'CovFatalities':[pcovariancefatalities],'CovCases':[pcovarianceCC]})
        optimalparamsAll = pd.concat([optimalparamsAll, optimalparams_add], ignore_index=True)
        
test_calc=test_calc.sort_index()
train_calc=train_calc.sort_index()

#Submit final results
test_calc.to_csv('submission.csv', index = False)
# %% [code]
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

# %% [code]
from pandas_profiling import ProfileReport

# %% [code]
# Load Data
xtrain = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
xtest = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
xsubmission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')

# %% [code]
train_profile = ProfileReport(xtrain, title='Pandas Profiling Report', html={'style':{'full_width':True}})
train_profile

# %% [code]
xtrain.rename(columns={'Country_Region':'Country'}, inplace=True)
xtest.rename(columns={'Country_Region':'Country'}, inplace=True)

xtrain.rename(columns={'Province_State':'State'}, inplace=True)
xtest.rename(columns={'Province_State':'State'}, inplace=True)

xtrain['Date'] = pd.to_datetime(xtrain['Date'], infer_datetime_format=True)
xtest['Date'] = pd.to_datetime(xtest['Date'], infer_datetime_format=True)

xtrain.info()
xtest.info()

y1_xTrain = xtrain.iloc[:, -2]
y1_xTrain.head()
y2_xTrain = xtrain.iloc[:, -1]
y2_xTrain.head()

EMPTY_VAL = "EMPTY_VAL"

def fillState(state, country):
    if state == EMPTY_VAL: return country
    return state

# %% [code]
X_xTrain = xtrain.copy()

X_xTrain['State'].fillna(EMPTY_VAL, inplace=True)
X_xTrain['State'] = X_xTrain.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

X_xTrain.loc[:, 'Date'] = X_xTrain.Date.dt.strftime("%m%d")
X_xTrain["Date"]  = X_xTrain["Date"].astype(int)

X_xTrain.head()

#X_Test = df_test.loc[:, ['State', 'Country', 'Date']]
X_xTest = xtest.copy()

X_xTest['State'].fillna(EMPTY_VAL, inplace=True)
X_xTest['State'] = X_xTest.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

X_xTest.loc[:, 'Date'] = X_xTest.Date.dt.strftime("%m%d")
X_xTest["Date"]  = X_xTest["Date"].astype(int)

X_xTest.head()

# %% [code]
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

X_xTrain.Country = le.fit_transform(X_xTrain.Country)
X_xTrain['State'] = le.fit_transform(X_xTrain['State'])

X_xTrain.head()

X_xTest.Country = le.fit_transform(X_xTest.Country)
X_xTest['State'] = le.fit_transform(X_xTest['State'])

X_xTest.head()

xtrain.head()
xtrain.loc[xtrain.Country == 'Afghanistan', :]
xtest.tail()

# %% [code]
from warnings import filterwarnings
filterwarnings('ignore')

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

from xgboost import XGBRegressor

countries = X_xTrain.Country.unique()


# %% [code]
# Predict data and Create submission file from test data
xout = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

for country in countries:
    states = X_xTrain.loc[X_xTrain.Country == country, :].State.unique()
    #print(country, states)
    # check whether string is nan or not
    for state in states:
        X_xTrain_CS = X_xTrain.loc[(X_xTrain.Country == country) & (X_xTrain.State == state), ['State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities']]
        
        y1_xTrain_CS = X_xTrain_CS.loc[:, 'ConfirmedCases']
        y2_xTrain_CS = X_xTrain_CS.loc[:, 'Fatalities']
        
        X_xTrain_CS = X_xTrain_CS.loc[:, ['State', 'Country', 'Date']]
        
        X_xTrain_CS.Country = le.fit_transform(X_xTrain_CS.Country)
        X_xTrain_CS['State'] = le.fit_transform(X_xTrain_CS['State'])
        
        X_xTest_CS = X_xTest.loc[(X_xTest.Country == country) & (X_xTest.State == state), ['State', 'Country', 'Date', 'ForecastId']]
        
        X_xTest_CS_Id = X_xTest_CS.loc[:, 'ForecastId']
        X_xTest_CS = X_xTest_CS.loc[:, ['State', 'Country', 'Date']]
        
        X_xTest_CS.Country = le.fit_transform(X_xTest_CS.Country)
        X_xTest_CS['State'] = le.fit_transform(X_xTest_CS['State'])
        
        #models_C[country] = gridSearchCV(model, X_Train_CS, y1_Train_CS, param_grid, 10, 'neg_mean_squared_error')
        #models_F[country] = gridSearchCV(model, X_Train_CS, y2_Train_CS, param_grid, 10, 'neg_mean_squared_error')
        
        xmodel1 = XGBRegressor(n_estimators=1000)
        xmodel1.fit(X_xTrain_CS, y1_xTrain_CS)
        y1_xpred = xmodel1.predict(X_xTest_CS)
        
        xmodel2 = XGBRegressor(n_estimators=1000)
        xmodel2.fit(X_xTrain_CS, y2_xTrain_CS)
        y2_xpred = xmodel2.predict(X_xTest_CS)
        
        xdata = pd.DataFrame({'ForecastId': X_xTest_CS_Id, 'ConfirmedCases': y1_xpred, 'Fatalities': y2_xpred})
        xout = pd.concat([xout, xdata], axis=0)


# %% [code]
xout.ForecastId = xout.ForecastId.astype('int')
xout.tail()
xout.to_csv('submission.csv', index=False)
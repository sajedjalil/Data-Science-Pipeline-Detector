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
import matplotlib.pyplot as plt


# %% [code]
train_df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
test_df = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
submission = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

# %% [code]
train_df.head()

# %% [code]
train_df.info()

# %% [code]
train_df['Province_State'].fillna('',inplace=True)

# %% [code]
train_df['Date']=pd.to_datetime(train_df['Date'])

# %% [code]
train_df.info()

# %% [code]
train_df['Country_Region'].nunique()

# %% [code]
print(train_df['Date'].sort_values())
print((train_df['Date'].max()-train_df['Date'].min()).days)
#The training data given is of 82 days from 2020-01-22 to 2020-04-13

# %% [code]
train_df['Fatalities'].sort_values(ascending = False)

# %% [code]
Fatal = pd.DataFrame
Fatal = train_df
Fatal

# %% [code]
Fatal.drop(['Id','Province_State','Date','ConfirmedCases'],axis=1)

# %% [code]
Sum = Fatal.groupby('Country_Region')['Fatalities'].sum().to_frame().reset_index().sort_values(by = 'Fatalities', ascending = False)
Sum

# %% [code]
High_Fatal=pd.DataFrame
High_Fatal = Sum.nlargest(10,['Fatalities'])
High_Fatal

# %% [code]
#top 40 countires having low fatalities count as of 2020-04-13
Small_Fatal = pd.DataFrame
Small_Fatal = Sum.nsmallest(40,'Fatalities')
Small_Fatal

# %% [code]
#top 10 fatalities recorded countries as of 2020-04-13
plt.figure(figsize=(50,40))

ax1 = plt.subplot(121, aspect='equal')

High_Fatal.plot(kind='pie', y = 'Fatalities', ax=ax1, autopct='%1.1f%%',startangle=90, shadow=False,labels=High_Fatal['Country_Region'], legend = True, fontsize=20) 

plt.show()

# %% [code]
Confirmed = pd.DataFrame
Confirmed = train_df
Confirmed

# %% [code]
Confirmed.drop(['Id','Province_State','Date','Fatalities'],axis=1)

# %% [code]
Sum1= Confirmed.groupby('Country_Region')['ConfirmedCases'].sum().to_frame().reset_index().sort_values(by = 'ConfirmedCases', ascending = False)
Sum1

# %% [code]
High_confirm=pd.DataFrame
High_confirm = Sum1.nlargest(10,['ConfirmedCases'])
High_confirm

# %% [code]
#top 10 confirmed cases record countries as of 2020-04-13
plt.figure(figsize=(50,40))

ax1 = plt.subplot(121, aspect='equal')

High_confirm.plot(kind='pie', y = 'ConfirmedCases', ax=ax1, autopct='%1.1f%%',startangle=90, shadow=False,labels=High_confirm['Country_Region'], legend = True, fontsize=20) 

plt.show()

# %% [code]
#top 40 countires having low confirmed cases count as of 2020-04-13
Small_confirm=pd.DataFrame
Small_confirm = Sum1.nsmallest(30,['ConfirmedCases'])
Small_confirm

# %% [code]
#Separating the date into day,month,year in train data
train_df['day'] = train_df['Date'].dt.day
train_df['month'] = train_df['Date'].dt.month
train_df['year'] = train_df['Date'].dt.year

# %% [code]
train_df

# %% [code]
X = train_df.drop(['Id','Date','year','Fatalities','ConfirmedCases'],axis=1)
Y = train_df.iloc[:,4:6]

# %% [code]
X

# %% [code]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

# %% [code]
X_train

# %% [code]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train['Province_State']=le.fit_transform(X_train['Province_State'])
X_train['Country_Region']=le.fit_transform(X_train['Country_Region'])
X_test['Province_State']=le.fit_transform(X_test['Province_State'])
X_test['Country_Region']=le.fit_transform(X_test['Country_Region'])
#test_df['Province_State'] = lb1.transform(test_df['Province_State'])

# %% [code]
X_train

# %% [code]
#X=X.drop(['Province_State','Country_Region'],axis=1)

# %% [code]
X.info(memory_usage='deep')

# %% [code]
Y.info()

# %% [code]


# %% [code]
X.info()

# %% [code]
#As we have two target variables Confirmed and Fatal cases using MultiOutputRegressor to train the model
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
max_depth = 30
regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
                                                          max_depth=max_depth,
                                                          random_state=0))
regr_multirf.fit(X_train, Y_train)

# Predict on new data
predict = regr_multirf.predict(X_test)

# %% [code]
from sklearn.metrics import accuracy_score,classification_report,mean_squared_error
from math import sqrt 
rms=sqrt(mean_squared_error(predict,Y_test))
rms

# %% [code]
#Prediting for future dates based on train model
test_df.info()
# %% [code]
test_df['Province_State'].fillna('',inplace=True)
test_df['Date']=pd.to_datetime(test_df['Date'])
# %% [code]
test_df['Country_Region'].nunique()
# %% [code]
test_df.info()
# %% [code]
print(test_df['Date'].sort_values())
print((test_df['Date'].max()-test_df['Date'].min()).days)
# %% [code]
#Separating the date into day,month,year in train data
test_df['day'] = test_df['Date'].dt.day
test_df['month'] = test_df['Date'].dt.month
test_df['year'] = test_df['Date'].dt.year
# %% [code]
test_df.head()
# %% [code]
test_df1=test_df.drop(['ForecastId','Date','year',],axis=1)
# %% [code]
test_df1.info()
# %% [code]
test_df1['Province_State']=le.fit_transform(test_df1['Province_State'])
test_df1['Country_Region']=le.fit_transform(test_df1['Country_Region'])
# %% [code]
test_df1
# %% [code]
pred = regr_multirf.predict(test_df1)
predicted = pd.DataFrame(data = pred , columns=['ConfirmedCases','Fatalities'])
# %% [code]
predicted['ForecastId'] = test_df['ForecastId']
# %% [code]
predicted = predicted[['ForecastId','ConfirmedCases','Fatalities']]
# %% [code]
#future predictions
predicted
# %% [code]
#RootMeanSquaredError
rms=sqrt(mean_squared_error(predict,Y_test))
# %% [code]
rms

predicted.to_csv("submission.csv" , index = False)

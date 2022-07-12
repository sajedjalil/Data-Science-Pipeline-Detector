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
%matplotlib inline

# %% [code]
train_df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
test_df = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
submission_df = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

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
display(train_df['Date'].sort_values())
display((train_df['Date'].max()-train_df['Date'].min()).days)
#The training data given is of 82 days from 2020-01-22 to 2020-04-13

# %% [code]
test_df.head()

# %% [code]
test_df.info()

# %% [code]
test_df['Province_State'].fillna('',inplace=True)

# %% [code]
test_df['Date']=pd.to_datetime(test_df['Date'])

# %% [code]
test_df['Country_Region'].nunique()

# %% [code]
test_df.info()

# %% [code]
display(test_df['Date'].sort_values())
display((test_df['Date'].max()-test_df['Date'].min()).days)

# %% [code]
submission_df

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
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
train_df['Country_Region'] = lb.fit_transform(train_df['Country_Region'])
test_df['Country_Region'] = lb.transform(test_df['Country_Region'])

lb1 = LabelEncoder()
train_df['Province_State'] = lb1.fit_transform(train_df['Province_State'])
test_df['Province_State'] = lb1.transform(test_df['Province_State'])

# %% [code]
display(train_df.info())
display(test_df.info())

# %% [code]
test_df.head()

# %% [code]
#Separating the date into day,month,year in train data
train_df['day'] = train_df['Date'].dt.day
train_df['month'] = train_df['Date'].dt.month
train_df['year'] = train_df['Date'].dt.year

# %% [code]
#Separating the date into day,month,year in train data
test_df['day'] = test_df['Date'].dt.day
test_df['month'] = test_df['Date'].dt.month
test_df['year'] = test_df['Date'].dt.year

# %% [code]


# %% [code]
train_df['ConfirmedCases'] = train_df['ConfirmedCases'].apply(int)
train_df['Fatalities'] = train_df['Fatalities'].apply(int)

# %% [code]
train_df.info()

# %% [code]
X = train_df.drop(['Id','Date','Fatalities','ConfirmedCases'],axis=1)
Y = train_df.drop(['Id','Province_State','Date','Country_Region','day','month','year'],axis=1)

# %% [code]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

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
Submission = pd.DataFrame(data=predict,columns = ['ConfirmedCases','Fatalities'])
Submission.sort_values(by = 'ConfirmedCases')
Submission['ForecastId'] = submission_df['ForecastId']

# %% [code]
Submission['ConfirmedCases'] = Submission['ConfirmedCases'].apply(int)
Submission['Fatalities'] = Submission['Fatalities'].apply(int)

# %% [code]
Submission = Submission[['ForecastId','ConfirmedCases','Fatalities' ]]
Submission.sort_values(by='Fatalities', ascending = False)

# %% [code]
from sklearn.metrics import accuracy_score,classification_report,mean_squared_error
from math import sqrt 
rms=sqrt(mean_squared_error(predict,Y_test))
rms

# %% [code]
Submission.to_csv("Submission.csv" , index = False)
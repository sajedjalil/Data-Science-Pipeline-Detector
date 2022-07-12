# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import necessary packages
import pandas as pd
import datetime as dt
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
path = '/kaggle/input/covid19-global-forecasting-week-2/'

# read train data
train = pd.read_csv(path+"train.csv")

# Change colum Date to datetime
train['Date'] = pd.to_datetime(train['Date'])

# origine of dates
date0 = dt.datetime.strptime('2020-01-22', '%Y-%m-%d')

# Calculate Time column
train['Time'] = train['Date']-date0
i = 0
n = len(train['Time'])

for i in range(n):
    train.loc[i, 'Time'] = train.loc[i, 'Time'].days
    
# fill column "Province_State" with value in  "Country_Region" if empty
for i in range(n):
    if pd.isnull(train.loc[i, "Province_State"]):
        train.loc[i, "Province_State"] = train.loc[i, "Country_Region"] + '_'
        
# read test data
test = pd.read_csv(path + "test.csv")

# Change colum Date to datetime
test['Date'] = pd.to_datetime(test['Date'])

# Calculate Time column
test['Time'] = test['Date']-date0
i = 0
n = len(test['Time'])

for i in range(n):
    test.loc[i, 'Time'] = test.loc[i, 'Time'].days

# fill column "Province_State" with value in  "Country_Region" if empty
for i in range(n):
    if pd.isnull(test.loc[i, "Province_State"]):
        test.loc[i, "Province_State"] = test.loc[i, "Country_Region"] + '_'

        
        
# for each Pprovince_State, find the first Time with non zero ConfirmedCases
temp = train.loc[train['ConfirmedCases'] > 0]
train_time_of_first_case = pd.DataFrame(
    temp.groupby(["Province_State"])['Time'].min())

# for each Pprovince_State, find the first Time with non zero Fatalities
temp = train.loc[train['Fatalities'] > 0]
train_time_of_first_fatality = pd.DataFrame(
    temp.groupby(["Province_State"])['Time'].min())


# prevision

# parameters
test["ConfirmedCases"] = 0
test["Fatalities"] = 0
a = 0.767
b = 0.31
c = 2.26
taux = 0

for i in range(0, n):
    t = test.loc[i, "Time"]
    province = test.loc[i, "Province_State"]

    if t >= 57 and t <= 65:
        temp = train.loc[train["Province_State"] == province]
        temp = temp.loc[temp["Time"] == t]
        test.loc[i, "ConfirmedCases"] = temp.iloc[0]['ConfirmedCases']
        test.loc[i, "Fatalities"] = temp.iloc[0]['Fatalities']
        continue

    # Confirmed cases
    if province in list(train_time_of_first_case.index):
        taux_case = a*(t-55)/(b*((t-55)**2)+c)
        test.loc[i, "ConfirmedCases"] = int(
            test.loc[i-1, "ConfirmedCases"] * (1+taux_case))
    else:
        test.loc[i, "ConfirmedCases"] = 0

    # Fatalities
    if province in list(train_time_of_first_fatality.index):
        taux_fat = a*(t-55)/(b*((t-55)**2)+c)
        test.loc[i, "Fatalities"] = int(
            test.loc[i-1, "Fatalities"] * (1+taux_fat))
    else:
        test.loc[i, "Fatalities"] = 0

# submission file
submission = test[['ForecastId', 'ConfirmedCases', 'Fatalities']]
submission.to_csv('submission.csv', index=False)






    
    
    
    
    



























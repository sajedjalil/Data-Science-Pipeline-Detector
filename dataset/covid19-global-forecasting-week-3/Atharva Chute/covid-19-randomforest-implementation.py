# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")

train.info()
test.info()

# Filling the nan columns with ""
train["Province_State"] = train["Province_State"].fillna('')
test["Province_State"] = test["Province_State"].fillna('')


# Spliting the "Date" column to "Month" and "Day" columns 
train["Month"], train["Day"] = 0, 0
for i in range(len(train)):
    train["Month"][i] = (train["Date"][i]).split("-")[1]
    train["Day"][i] = (train["Date"][i]).split("-")[2]
    
test["Month"], test["Day"] = 0, 0
for i in range(len(test)):
    test["Month"][i] = (test["Date"][i]).split("-")[1]
    test["Day"][i] = (test["Date"][i]).split("-")[2]
    
countries = train["Country_Region"].unique()
states = train["Province_State"].unique()


# Creating the "Country/State" column which contains the essence of "Province_State" and "Country_Region" columns
for i in range(len(train)):
    if train["Province_State"][i] != '':
        train["Country_Region"][i] = train["Province_State"][i] + " (" + str(train["Country_Region"][i]) + ")"
       
for i in range(len(test)):
    if test["Province_State"][i] != '':
        test["Country_Region"][i] = test["Province_State"][i] + " (" + str(test["Country_Region"][i]) + ")"

        
train.drop(columns = "Province_State", inplace=True)
test.drop(columns = "Province_State", inplace=True)

train.rename(columns = {"Country_Region" : "Country/State"}, inplace=True)
test.rename(columns = {"Country_Region" : "Country/State"}, inplace=True)


# Converting the "Day" column to "# of Days"
i = 0
for value in train["Country/State"].unique():
    if i < len(train):
        j = 1
        while(train["Country/State"][i] == value):
            train["Day"][i] = j
            j += 1; i += 1
            if i == len(train):
                break
i = 0
for value in test["Country/State"].unique():
    if i < len(test):
        j = 65
        while(test["Country/State"][i] == value):
            test["Day"][i] = j
            j += 1; i += 1
            if i == len(test):
                break
train.rename(columns = {"Day" : "# of Days"}, inplace=True)
test.rename(columns = {"Day" : "# of Days"}, inplace=True)

# Dropping columns not required or stored in other variables
train_id = train["Id"].copy()
train_cc = train["ConfirmedCases"].copy()
train_ft = train["Fatalities"].copy()
test_id = test["ForecastId"].copy()

train = train.drop(columns = ["Date", "Id", "Fatalities", "ConfirmedCases"])
test = test.drop(columns = ["Date", "ForecastId"])

countriesorstates = train["Country/State"].unique()


# Binary Encoding to all the categorical features   
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()

lb.fit(train.iloc[:, 0])
dummy = lb.transform(train.iloc[:, 0])
dummy = np.delete(dummy, len(dummy.T)-1, 1)
dummy = pd.DataFrame(dummy)
train = pd.concat([train, dummy], axis = 1)
dummy = lb.transform(test.iloc[:, 0])
dummy = np.delete(dummy, len(dummy.T)-1, 1)    
dummy = pd.DataFrame(dummy)
test = pd.concat([test, dummy], axis = 1)


train = train.drop(columns = ["Country/State"])
test = test.drop(columns = ["Country/State"])

columns = train.columns

# Just changing column names
for i in range(len(train.T)):
    train.rename(columns = {columns[i] : "Feature "+str(i)}, inplace=True)
    
for i in range(len(test.T)):
    test.rename(columns = {columns[i] : "Feature "+str(i)}, inplace=True)
    
tr = train.copy()
tt = test.copy()

# Scaling the numeric columns
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
tr.iloc[:, 0:2] = sc.fit_transform(tr.iloc[:, 0:2])
tt.iloc[:, 0:2] = sc.transform(tt.iloc[:, 0:2])

train_cc = train_cc.to_frame()
train_ft = train_ft.to_frame()
tr_cc = train_cc.copy()
tr_ft = train_ft.copy()

# Omitting the duplicated columns
tr = tr.loc[:, ~tr.columns.duplicated()]
"""duplicated_columnstr = tr.columns[tr.columns.duplicated()]"""

tt = tt.loc[:, ~tt.columns.duplicated()]
"""duplicated_columnstt = tt.columns[tt.columns.duplicated()]"""

# Applying the model
from sklearn.ensemble import RandomForestRegressor
reg_cc = RandomForestRegressor(n_estimators = 1000, random_state = 0)

tr_cc = sc.fit_transform(tr_cc)
reg_cc.fit(tr, tr_cc)
tt_cc = reg_cc.predict(tt)
tt_cc = sc.inverse_transform(tt_cc)

reg_ft = RandomForestRegressor(n_estimators = 1000, random_state = 0)

tr_ft = sc.fit_transform(tr_ft)
reg_ft.fit(tr, tr_ft)
tt_ft = reg_ft.predict(tt)
tt_ft = sc.inverse_transform(tt_ft)

# Converting to int
for i in range(len(tt_cc)):
    tt_cc[i] = int(round(tt_cc[i]))
    tt_ft[i] = int(round(tt_ft[i]))

tt_cc = pd.DataFrame(tt_cc)
tt_ft = pd.DataFrame(tt_ft)
train_id = train_id.to_frame()

sub = pd.DataFrame()
sub["ForecastId"] = test_id
sub["ConfirmedCases"] = tt_cc
sub["Fatalities"] = tt_ft

# saving the results
sub.to_csv("submission.csv", index = False)
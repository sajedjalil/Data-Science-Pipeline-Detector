# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Data input including 
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")
test =  pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")
submission =  pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/submission.csv")
country = pd.read_csv("/kaggle/input/covid19inf/covid19countryinfo.csv")

#Data munging
country.drop(columns= country.columns[range(22,54)], inplace=True)
country["pop"] = country["pop"].str.replace(",", "").astype('float64')
country["quarantine"] = pd.to_datetime(country.quarantine)
country["schools"] = pd.to_datetime(country.schools)
country["restrictions"] = pd.to_datetime(country.restrictions)

train["Date"] = pd.to_datetime(train.Date)
train["province"] = train["Province_State"]
train.province.fillna(train["Country_Region"], inplace=True)
test["Date"] = pd.to_datetime(test.Date)
test["province"] = test["Province_State"]
test.province.fillna(test["Country_Region"], inplace=True)
train = train.merge(country, how='left', left_on = ["province"], right_on = ["country"])
test = test.merge(country, how='left', left_on = ["province"], right_on = ["country"])
train["days"] = (train.Date - pd.to_datetime("2020-01-22")).dt.days
test["days"] = (test.Date - pd.to_datetime("2020-01-22")).dt.days
#Construction of laged variables 
lag_number = 3
for lag in range(1, lag_number + 1):
    var_name = "cases_lag%d" % lag
    train[var_name] = train.ConfirmedCases.shift(periods = 1)
    train.loc[train.Date <= train.Date[lag - 1] , var_name] = 0
    var_name = "fatalities_lag%d" % lag
    train[var_name] = train.Fatalities.shift(periods = 1)
    train.loc[train.Date <= train.Date[lag - 1] , var_name] = 0
    
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
train["province"] = lb.fit_transform(train.province)
test["province"] = lb.transform(test.province)

train["days_from_first_case"] = -1
test["days_from_first_case"] = -1
#train["days_from_first_death"] = -1
train["days_from_case_100"] = -1
test["days_from_case_100"] = -1

dates = list(train.Date.unique())
for province in train.province.unique():
    #print(province)
    mask1 = train.province == province
    mask2 = train.ConfirmedCases > 0.0
    mask3 = train.ConfirmedCases > 100.0
    try:
        idx1 = train.loc[mask1 & mask2 ,["ConfirmedCases"]].idxmin()[0]
        dateidx1 = train.iloc[idx1]["Date"]
    except:
        dateidx1 = train.Date.max()
        pass
    #print(dateidx1)
    train.loc[mask1 & (train.Date >= dateidx1), "days_from_first_case"] = (train.Date - dateidx1).dt.days
    test.loc[mask1 & (test.Date >= dateidx1), "days_from_first_case"] = (test.Date - dateidx1).dt.days
    
    try:
        idx1 = train.loc[mask1 & mask3 ,["ConfirmedCases"]].idxmin()[0]
        dateidx1 = train.iloc[idx1]["Date"]
    except:
        dateidx1 = train.Date.max()
        pass
    train.loc[mask1 & (train.Date >= dateidx1), "days_from_case_100"] = (train.Date - dateidx1).dt.days
    test.loc[mask1 & (test.Date >= dateidx1), "days_from_case_100"] = (test.Date - dateidx1).dt.days    

    #for date in dates:
    #    print(date)
    #    train.loc[train.province == province & train.Date == date & train.ConfirmedCases>0, "ConfirmedCases"].min()
train.fillna(value = 1, inplace = True)
test.fillna(value = 1, inplace = True)

X_train_cases = train.drop(columns = ["Id", "Province_State", "Country_Region", "Date", "ConfirmedCases", "Fatalities", 
                                      "country", "quarantine" , "schools", "restrictions"])
y_train_cases = train.ConfirmedCases

regr_cases = RandomForestRegressor(n_estimators= 400, max_depth=5, random_state=0, verbose=0)
regr_cases.fit(X_train_cases, y_train_cases)
predict_train_cases = regr_cases.predict(X_train_cases)
print("RMSE in detected cases: ", np.sqrt(mean_squared_error(y_train_cases, predict_train_cases)))

X_train_fatalities = train.drop(columns = ["Id", "Province_State", "Country_Region", "Date", "Fatalities", 
                                      "country", "quarantine" , "schools", "restrictions"])
y_train_fatalities = train.Fatalities

regr_fatalities = RandomForestRegressor(n_estimators= 400, max_depth=5, random_state=0, verbose=0)
regr_fatalities.fit(X_train_fatalities, y_train_fatalities)
predict_train_fatalities = regr_fatalities.predict(X_train_fatalities)
print("RMSE in fatalities: ", np.sqrt(mean_squared_error(y_train_fatalities, predict_train_fatalities)))

X_test_cases = test.drop(columns = ["ForecastId", "Province_State", "Country_Region", "Date", 
                                    "country", "quarantine" , "schools", "restrictions"])

lags = {}
predict_test_cases, predict_test_fatalities = [] , []
test_min_day = test.days.min()
for i in range(1, lag_number + 1):
    lags["caseslag%d" % i] = 0
    lags["fatalitieslag%d" % i] = 0
    
for ind in range(len(X_test_cases)):
    print("case: {} of {}".format(ind, len(X_test_cases)))
    #First lag data are obtained either from previous calculations or train data
    if X_test_cases.iloc[ind].days == test_min_day:
        print(test_min_day)
        for i in range(1, lag_number + 1):
            mask1 = train.days == (test_min_day - i)
            mask2 = train.province == X_test_cases.iloc[ind].province
            lags["caseslag%d" % i] = train.loc[mask1 & mask2, "ConfirmedCases"].values[0]
            lags["fatalitieslag%d" % i] = train.loc[mask1 & mask2, "Fatalities"].values[0]
    else:
        lags["caseslag1"] = pred_cases
        lags["fatalitieslag1"] = pred_fatalities
        for i in range(2, lag_number + 1):
            lags["caseslag%d" % i] = lags["caseslag%d" % (i-1)]
            lags["fatalitieslag%d" % i] = lags["fatalitieslag%d" % (i-1)]
    x_test = X_test_cases.iloc[ind].copy()
    for i in range(1, lag_number + 1):
        x_test["cases_lag%d" % i] = lags["caseslag%d" % i]
        x_test["fatalities_lag%d" % i] = lags["fatalitieslag%d" % i]
        
    x_test =pd.DataFrame(x_test).transpose()
    pred_cases = regr_cases.predict(x_test)[0]
    x_test["ConfirmedCases"] = pred_cases
    pred_fatalities = regr_fatalities.predict(x_test)[0]
    predict_test_cases.append(pred_cases)
    predict_test_fatalities.append(pred_fatalities)

submission.ConfirmedCases = predict_test_cases
submission.Fatalities = predict_test_fatalities
submission.to_csv("submission.csv", index = False)





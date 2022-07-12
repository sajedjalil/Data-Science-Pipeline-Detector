# COVID 19 Forecast Week 2 using Decision Tree Regressor

# Importing the libraries
import pandas as pd
import numpy as np

# Importing the datasets
dataset = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')
ds = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')

# Filling missing Province values with the corresponding country
dataset['Province_State'] = dataset['Province_State'].fillna(dataset['Country_Region'])
ds['Province_State'] = ds['Province_State'].fillna(ds['Country_Region'])

# Extracting Month, Day & Time Stamp from Date column
from datetime import datetime

month = lambda x: datetime.strptime(x, '%Y-%m-%d').month
day = lambda x: datetime.strptime(x, '%Y-%m-%d').day

dataset['Month'] = dataset['Date'].map(month)
dataset['Day'] = dataset['Date'].map(day)

dataset['Date'] = dataset['Date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
dataset['Timestamp'] = dataset['Date'].apply(lambda x: x.timestamp()).astype(int)

ds['Month'] = ds['Date'].map(month)
ds['Day'] = ds['Date'].map(day)

ds['Date'] = ds['Date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
ds['Timestamp'] = ds['Date'].apply(lambda x: x.timestamp()).astype(int)


X = dataset.iloc[:, [1, 2, 6, 7, 8]].values
y = dataset.iloc[:,[4, 5]].values
Xx = ds.iloc[:, [1, 2, 4, 5, 6]].values

# OHE the Province & Country columns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

lbl = LabelEncoder()
X[:, 0] = lbl.fit_transform(X[:, 0])
X[:, 1] = lbl.fit_transform(X[:, 1])

Xx[:, 0] = lbl.fit_transform(Xx[:, 0])
Xx[:, 1] = lbl.fit_transform(Xx[:, 1])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
    remainder='passthrough')

X = ct.fit_transform(X).toarray()
Xx = ct.fit_transform(Xx).toarray()

# Building the DT Regression Model - Testing with known data
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(min_samples_split = 2, min_samples_leaf = 2,random_state = 1)
regressor.fit(X, y)

# Predictions
y_pred = pd.DataFrame(regressor.predict(Xx)).astype(int)

#y_pred_0 = y_pred.iloc[:, 0].astype(int)
#y_pred_1 = y_pred.iloc[:, 1].astype(int)

# Kaggle Submission
pd.DataFrame({"ForecastId":list(range(1,len(ds)+1)),
              "ConfirmedCases":y_pred.iloc[:,0],
              "Fatalities": y_pred.iloc[:,1]}).to_csv("submission.csv",
                                           index=False,
                                           header=True)

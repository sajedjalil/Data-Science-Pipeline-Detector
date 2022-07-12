# Importing the libraries
import pandas as pd
import numpy as np

# Importing the Keras libraries
from xgboost import XGBRegressor

# Importing datasets
dataset = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')
ds = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

dataset['Province_State'] = dataset['Province_State'] .fillna(dataset['Country_Region'])
ds['Province_State'] = ds['Province_State'] .fillna(ds['Country_Region'])

# Separating Day & Month into individual columns
from datetime import datetime
month = lambda x: datetime.strptime(x, "%Y-%m-%d").month
day = lambda x: datetime.strptime(x, "%Y-%m-%d").day

dataset['Month'] = dataset['Date'].map(month)
dataset['Day'] = dataset['Date'].map(day)

ds['Month'] = ds['Date'].map(month)
ds['Day'] = ds['Date'].map(day)

# Finally dividing the dataset into matrix of independent features - X
# and matrix of dependent variables - y
X = dataset.iloc[:,[1, 2, 5, 6]].values
y_cc = dataset.iloc[:,[4]].values
y_ft = dataset.iloc[:,[5]].values


Xx = ds.iloc[:,[1, 2, 4, 5]].values

# OHE
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

X = X[:, 1:]
Xx = Xx[:, 1:]

# Scaling - Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
Xx = sc.transform(Xx)

# RNN function
regressor = XGBRegressor()
regressor.fit(X, y_cc)

reg = XGBRegressor()
reg.fit(X, y_ft)

# Predictions
y_pred_cc = regressor.predict(Xx).astype(int)
y_pred_ft = reg.predict(Xx).astype(int)

# Kaggle Submission CSV
pd.DataFrame({"ForecastId":list(range(1,len(ds)+1)),
              "ConfirmedCases": y_pred_cc,
              "Fatalities": y_pred_ft}).to_csv("submission.csv",
                                           index=False,
                                           header=True)
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import tqdm
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from itertools import combinations
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge ,LinearRegression


province_encoder = LabelEncoder()
country_encoder = LabelEncoder()

import math


def pre_process(data, province_encoder, country_encoder):
    data['Province_State'] = data['Province_State'].replace({np.NaN: 'UNK'})
    data['month'] = data['Date'].apply(lambda x: int(x.split('-')[1]))
    data['day'] = data['Date'].apply(lambda x: int(x.split('-')[2]))
    data = data.drop('Date', axis=1)
    data['Province_State'] = province_encoder.transform(data['Province_State'].values)
    data['Country_Region'] = country_encoder.transform(data['Country_Region'].values)

    return data


def group_data(data, degree=3, hash=hash):
    new_data = []
    m, n = data.shape
    for indices in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:, indices]])
    return np.array(new_data).T


scaler = MinMaxScaler()

data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
data['Province_State'] = data['Province_State'].replace({np.NaN: 'UNK'})
province_encoder.fit(data['Province_State'].values)
country_encoder.fit(data['Country_Region'].values)

# preparing training data
data = pre_process(data, province_encoder, country_encoder)
test = pre_process(test, province_encoder, country_encoder)

data = data.set_index('Id')
test = test.set_index('ForecastId')


test['ConfirmedCases'] = np.zeros(len(test))
test['Fatalities'] = np.zeros(len(test))
cc, ft = data['ConfirmedCases'].values, data['Fatalities'].values
data = data.drop(['ConfirmedCases', 'Fatalities'], axis=1)


for val in data['Country_Region'].unique():

    country_mask = data['Country_Region'] == val
    X = data[country_mask]
    X = scaler.fit_transform(X)
    y1 = cc[country_mask]
    y2 = ft[country_mask]

    model1 = LinearRegression()
    model1.fit(X, y1)

    model2 = LinearRegression()
    model2.fit(X, y2)
    
    test_country_mask = test['Country_Region'] == val
    X_test = test[test_country_mask][['Province_State', 'Country_Region', 'month', 'day']]
    X_test = scaler.transform(X_test)
    
    test.loc[test_country_mask, 'ConfirmedCases'] = model1.predict(X_test)
    test.loc[test_country_mask, 'Fatalities'] = model2.predict(X_test)
    
test[test['Fatalities'] < 0] = 0
test[test['ConfirmedCases'] < 0] = 0
test = test.sort_index()
test = test.reset_index()
test = test[['ForecastId', 'ConfirmedCases', 'Fatalities']]
test.to_csv('submission.csv', index=False)



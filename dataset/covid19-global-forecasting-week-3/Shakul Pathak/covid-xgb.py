# %% [code]
import numpy as np
import pandas as pd

# %% [code]
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

# %% [code]
x_train = train
x_test = test

# %% [code]
x_train['Date'] = pd.to_datetime(x_train['Date'])
x_train['Date'] = x_train['Date'].dt.strftime("%m%d")
x_test['Date'] = pd.to_datetime(x_test['Date'])
x_test['Date'] = x_test['Date'].dt.strftime("%m%d")

# %% [code]
x_train = x_train.fillna('NA')
x_test = x_test.fillna('NA')

# %% [code]


# %% [code]
country_list = x_train['Country_Region'].unique()


# %% [code] {"scrolled":true}
from warnings import filterwarnings

filterwarnings('ignore')

from sklearn import preprocessing

from xgboost import XGBRegressor

encoder = preprocessing.LabelEncoder()

sub = []
for country in country_list:
    province_list = x_train.loc[x_train['Country_Region'] == country].Province_State.unique()
    for province in province_list:
        X_train = x_train.loc[(x_train['Country_Region'] == country) & (x_train['Province_State'] == province),['Date']].astype('int')
        Y_train_c = x_train.loc[(x_train['Country_Region'] == country) & (x_train['Province_State'] == province),['ConfirmedCases']]
        Y_train_f = x_train.loc[(x_train['Country_Region'] == country) & (x_train['Province_State'] == province),['Fatalities']]
        X_test = x_test.loc[(x_test['Country_Region'] == country) & (x_test['Province_State'] == province), ['Date']].astype('int')
        X_forecastId = x_test.loc[(x_test['Country_Region'] == country) & (x_test['Province_State'] == province), ['ForecastId']]
        X_forecastId = X_forecastId.values.tolist()
        X_forecastId = [v[0] for v in X_forecastId]
        model_c = XGBRegressor(n_estimators=1000)
        model_c.fit(X_train, Y_train_c)
        Y_pred_c = model_c.predict(X_test)
        model_f = XGBRegressor(n_estimators=1000)
        model_f.fit(X_train, Y_train_f)
        Y_pred_f = model_f.predict(X_test)
        for j in range(len(Y_pred_c)):
            dic = { 'ForecastId': X_forecastId[j], 'ConfirmedCases': Y_pred_c[j], 'Fatalities': Y_pred_f[j]}
            sub.append(dic)

# %% [code]
submission = pd.DataFrame(sub)
submission[['ForecastId','ConfirmedCases','Fatalities']].to_csv(path_or_buf='submission.csv',index=False)

# %% [code]


# %% [code]

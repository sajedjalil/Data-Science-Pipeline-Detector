# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
DATA_FOLDER = '../input/covid19-global-forecasting-week-1/'

def read_data_training():
  df_train = pd.read_csv(DATA_FOLDER + 'train.csv')
  df_train['Date'] = pd.to_datetime(df_train['Date'], format='%Y-%m-%d')
  return df_train

def print_all_rows(df):
  pd.set_option('display.max_rows', df.shape[0]+1)
  print(df)
   
def prophet_calculation(df_train, label):
  from fbprophet import Prophet
  selected_df_train = df_train[['Date', label]]
  selected_df_train.reset_index(inplace=True)
  selected_df_train.rename(columns={'Date': 'ds'}, inplace=True)

  # difference between two consecutive days
  selected_df_train['y'] = selected_df_train[label] - selected_df_train[label].shift(1)
  selected_df_train.loc[0, 'y'] = 0

  # Train the model
  model = Prophet(growth='linear')
  model.fit(selected_df_train)

  # Predict the change (difference between two consecutive days)
  future_df = model.make_future_dataframe(periods=50)
  forecast = model.predict(future_df)
  forecast['old'] = selected_df_train[label] # 'old' is old value of 'label' (confirmed cases or fatalities)
  forecast['yhat'] = pd.to_numeric(forecast['yhat'])
  forecast.loc[forecast['yhat'] < 0, ['yhat']] = 0
  # Ensure that 'yhat' is always increasing
  for i in range(1, len(forecast)):
    if forecast.loc[i - 1, 'yhat'] > forecast.loc[i, 'yhat']:
      forecast.loc[i, 'yhat'] = forecast.loc[i - 1, 'yhat']
  # Lower 'yhat' value by subtracting to the min value
#   min_value = forecast['yhat'].min()
#   forecast['yhat'] = forecast['yhat'].apply(lambda x: x - min_value)

  # Get prediction of 'label' (confirmed cases or fatalities)
  for i in range(len(forecast)):
    if not math.isnan(forecast.loc[i, 'old']):
      forecast.loc[i, label] = forecast.loc[i, 'old']
    else:
      forecast.loc[i, label] = forecast.loc[i - 1, label] + forecast.loc[i, 'yhat']

  forecast = forecast[['ds', label]]
  forecast = forecast.round({label: 0})
  forecast.rename(columns={'ds': 'Date'}, inplace=True)
  forecast = forecast.set_index('Date')
  return forecast

if __name__ == "__main__":
  # Create submission dataframe
  submission_df = pd.DataFrame(columns=['ForecastId', 'ConfirmedCases', 'Fatalities'])

  # Read test file
  df_test = pd.read_csv(DATA_FOLDER + 'test.csv')
  df_test['Date'] = pd.to_datetime(df_test['Date'], format='%Y-%m-%d')

  # Read data to train
  df_train = read_data_training()
  coordinates = list(zip(df_train['Lat'], df_train['Long'], df_train['Country/Region']))
  coordinates = list(dict.fromkeys(coordinates))
  # count = 0
  for coordinate in coordinates:
    # print(coordinate)
    df_train_region = df_train[((df_train['Lat'] == coordinate[0]) & (df_train['Long'] == coordinate[1]) & (df_train['Country/Region'] == coordinate[2]))]
    df_test_region = df_test[((df_test['Lat'] == coordinate[0]) & (df_test['Long'] == coordinate[1]) & (df_test['Country/Region'] == coordinate[2]))]
    df_test_region = df_test_region.set_index('Date')

    forecast_case = prophet_calculation(df_train_region, "ConfirmedCases")
    forecast_fatalities = prophet_calculation(df_train_region, "Fatalities")
    df_test_region['ConfirmedCases'] = forecast_case['ConfirmedCases']
    df_test_region['Fatalities'] = forecast_fatalities['Fatalities']
    df_test_region = df_test_region[['ForecastId', 'ConfirmedCases', 'Fatalities']]
    df_test_region['ConfirmedCases'] = df_test_region['ConfirmedCases'].astype(np.int64)
    df_test_region['Fatalities'] = df_test_region['Fatalities'].astype(np.int64)
    # print_all_rows(df_test_region)
    submission_df = submission_df.append(df_test_region, ignore_index=True)
    # if count == 1:
    #   break
    # count += 1
  submission_df.to_csv('submission.csv', index=False)

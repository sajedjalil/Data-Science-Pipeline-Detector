# Project/Competition: https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/
# Basic prophet prediction

import pandas as pd
import numpy as np
import ProphetModeller # a custom class - see its source at https://gist.github.com/gvyshnya/07b0444104cf6021e9756b2e88df76d0
import datetime as dt

start_time = dt.datetime.now()
print("Started at ", start_time)

debug = 1

# read the training data.

print("Reading input data into memory")
date_info = pd.read_csv('../input/date_info.csv', parse_dates=['calendar_date'])
air_visit_data = pd.read_csv('../input/air_visit_data.csv', parse_dates=['visit_date'])

air_store_info = pd.read_csv('../input/air_store_info.csv')

if debug:
    print("Head of training data...")
    print(air_visit_data.head())

store_codes = ['air_4ce7b17062a1bf73',
                'air_310e467e6e625004'] # TODO: obtain the full list of store IDs at this step
if debug:
    print("Output air store codes")
    print(store_codes)

# forecasting parameters and structures
N = 39 # forecasting period: 8+31 days - from Apr 23, 2017 to May 31, 2017 inclusive
df_forecast = pd.DataFrame(columns=['id', 'visitors']) # the final DF with forecasted visits
store_forecasts = [] # the list of dataframes with structure similar to df_forecast

for i in range(len(store_codes)):

    current_store_id = store_codes[i]

    # get data for shop visits in the past
    query_str = 'air_store_id == "' + current_store_id + '"'
    data = air_visit_data.query(query_str)

    # prophet expects the  label names to be 'ds' (date and time in ts) and 'y' (value)
    X = pd.DataFrame(index=range(0,len(data)))
    X['ds'] = data['visit_date'].values
    X['y'] = data['visitors'].values
    if debug:
        print("Ready to forecast with prophet-ready data:")
        print(X.tail())

    # prepare forecast

    if np.mean(X['y'].values) == 0:
        print("Input for Prophet is ts with zero values, store #", i, ", store_id: ", current_store_id)
        # We do not collect a forecast here - missing values will be replaced by 0 visits in
        # 'Prepare submission dataframe...' section, anyway
    else:

        # setup Prophet Modeller
        do_log_transform = 1
        future_periods = N
        prophet = ProphetModeller.ProphetModeller(X, future_periods, do_log_transform)
        prophet.weekly_seasonality = True
        prophet.yearly_seasonality = True
        prophet.frequency = 'D'
        prophet.turn_negative_forecasts_to_0 = 1
        print("Started prophet model predictions at ", dt.datetime.now())
        prophet.predict()
        print("Finished prophet model predictions at ", dt.datetime.now())
        df_current_forecast = prophet.get_forecast_only()

        if debug:
            print("Forecasted visits, store #", i, ", store id: ", current_store_id)
            print(df_current_forecast.head())

        # build a submission-ready dataframe
        visit_ids = current_store_id + '_' + df_current_forecast['ds'].astype(str)
        visit_predictions = df_current_forecast['yhat'].values

        df_current_store_forecast = pd.DataFrame(columns=['id', 'visitors'])
        df_current_store_forecast['id'] = visit_ids
        df_current_store_forecast['visitors'] = visit_predictions

        store_forecasts.append(df_current_store_forecast)


print("Merge prediction dataframes into one final dataframe ...")

# combine (bind) individual predictions DFs into a single one
for i in range(len(store_forecasts)):
    if i == 0:
        df_forecast = store_forecasts[i]
    else:
        df_forecast = pd.concat([df_forecast, store_forecasts[i]])

# if for any reason the negative visits are forecasted, replace them with 0
df_forecast.loc[df_forecast.visitors < 0, 'visitors'] = 0

print('Output submission dataframe...')

df_forecast.to_csv('prophet.csv', index=False, float_format='%.4f')

print('We are done. That is all, folks!')
finish_time = dt.datetime.now()
print("Started at ", finish_time)
elapsed = finish_time - start_time
print("Elapsed time: ", elapsed)
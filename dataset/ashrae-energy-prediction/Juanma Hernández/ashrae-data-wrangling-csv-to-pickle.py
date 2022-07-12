# !/usr/bin/env python
"""CSV reading is very time consuming.
The goal of this script is to do this step once and use it in other scripts or notebooks.

The data is imported and transformed adequately to occupy less memory and to be more convenient to work with.

Te data is imported as the following types:

- `building_id` as category
- `meter` as category
- `meter_reading` as float 32
- 'row_id' as uint32
- `timestamp` as datatime with `%Y-%m-%d %H:%M:%S` format

The `meter` categories are renamed to:

- electricity
- chilledwater
- steam
- hotwater

to be more human readable and because it appears with the name in the graphs and legends.

The X_train is reconstituted in a new DataFrame, filling the gaps with NaN.
This method has a problem, not all the buildings has the four energy aspects.
Some building have only one, two or three energy aspects.
And the reconstitution, recreates all four energy aspects.
These non-existing energy aspects are removed.

"""

# !pip install nb_black
# %load_ext nb_black
import gc
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

__author__ = "Juanma Hernández"
__copyright__ = "Copyright 2019"
__credits__ = ["Juanma Hernández"]
__license__ = "GPL"
__maintainer__ = "Juanma Hernández"
__email__ = "https://twitter.com/juanmah"
__status__ = "Data wrangling"

print("> Prepare data")
data_path = "../input/ashrae-energy-prediction/"
date_parser = lambda x: pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S")
dtype = {
    "building_id": "category",
    "meter": "category",
    "meter_reading": "float32",
    "row_id": "uint32",
    "site_id": "category",
    "primary_use": "category",
    "square_feet": "uint32",
    "year_built": "Int16",
    "floor_count": "Int8",
    "air_temperature": "float32",
    "cloud_coverage": "Int8",
    "dew_temperature": "float32",
    "precip_depth_1_hr": "Int16",
    "sea_level_pressure": "float32",
    "wind_direction": "Int16",
    "wind_speed": "float32",
}

print('  Train set')
X_train = pd.read_csv(data_path + 'train.csv',
                      engine='c',
                      dtype=dtype,
                      parse_dates=['timestamp'],
                      date_parser=date_parser)
X_train['meter'].cat.rename_categories({'0': 'electricity',
                                        '1': 'chilledwater',
                                        '2': 'steam',
                                        '3': 'hotwater'},
                                       inplace=True)
X_train.head()

print('  Test set')
X_test = pd.read_csv(data_path + 'test.csv',
                     engine='c',
                     dtype=dtype,
                     parse_dates=['timestamp'],
                     date_parser=date_parser)
X_test['meter'].cat.rename_categories({'0': 'electricity',
                                       '1': 'chilledwater',
                                       '2': 'steam',
                                       '3': 'hotwater'},
                                      inplace=True)
X_test.head()

print("  Building metadadata")
building_metadata = pd.read_csv(
    data_path + "building_metadata.csv", engine="c", dtype=dtype
)
building_metadata.head()

print("  Weather train")
weather_train = pd.read_csv(
    data_path + "weather_train.csv",
    engine="c",
    dtype=dtype,
    parse_dates=["timestamp"],
    date_parser=date_parser,
)
weather_train.head()

print("  Weather test")
weather_test = pd.read_csv(
    data_path + "weather_test.csv",
    engine="c",
    dtype=dtype,
    parse_dates=["timestamp"],
    date_parser=date_parser,
)
weather_test.head()

print('> Process weather')
print('  Join train and set')
weather = pd.concat([weather_train, weather_test])
print('  Regenerate non existing timestamps')
weather = weather.groupby(["timestamp", "site_id"]).agg([np.sum])
weather.columns = weather.columns.droplevel(1)
weather.rename_axis(None, axis="columns")
weather.reset_index(inplace=True)
weather["isNaN"] = weather["air_temperature"].isna().astype("int")
weather.head()
weather.shape
print('  Apply time zone')
site_id_to_time_zone = {
    '0': "US/Eastern",
    '1': "Europe/London",
    '2': "US/Arizona",
    '3': "US/Eastern",
    '4': "US/Pacific",
    '5': "Europe/London",
    '6': "US/Eastern",
    '7': "Canada/Eastern",
    '8': "US/Eastern",
    '9': "US/Central",
    '10': "US/Pacific",
    '11': "Canada/Eastern",
    '12': "Europe/Dublin",
    '13': "US/Central",
    '14': "US/Eastern",
    '15': "US/Eastern",
}
weather["time_zone"] = weather["site_id"].map(site_id_to_time_zone)
weather.rename(columns={"timestamp": "timestamp_utc"}, inplace=True)
weather["timestamp"] = weather.apply(
    lambda x: x["timestamp_utc"]
    .tz_localize("UTC")
    .tz_convert(x["time_zone"])
    .tz_localize(None),
    axis="columns",
)
print('  Split train and set')
weather_train = weather[(weather['timestamp'] < '2017-01-01 00:00:00') & (weather['timestamp'] >= '2016-01-01 00:00:00')]
weather_test = weather[weather['timestamp'] >= '2017-01-01 00:00:00']

print("> Export data")
print('  Train set')
with open("X_train.pickle", "wb") as fp:
    pickle.dump(X_train, fp)
print('  Test set')
with open("X_test.pickle", "wb") as fp:
    pickle.dump(X_test, fp)
    X_test = None
print("  Building metadadata")
with open("building_metadata.pickle", "wb") as fp:
    pickle.dump(building_metadata, fp)
    building_metadata = None
print("  Weather train")
with open("weather_train.pickle", "wb") as fp:
    pickle.dump(weather_train, fp)
    weather_train = None
print("  Weather test")
with open("weather_test.pickle", "wb") as fp:
    pickle.dump(weather_test, fp)
    weather_test = None

print('> NaN')
gc.collect()
print('  Calculate')
print('  Q1')
# nan = X_train[0:10000].groupby(['timestamp', 'building_id', 'meter']).agg([sum])
nan1 = X_train[0:4779153].groupby(['timestamp', 'building_id', 'meter']).agg([sum])
print('  Q2')
nan2 = X_train[4779153:9898185].groupby(['timestamp', 'building_id', 'meter']).agg([sum])
print('  Q3')
nan3 = X_train[9898185:15052908].groupby(['timestamp', 'building_id', 'meter']).agg([sum])
print('  Q4')
nan4 = X_train[15052908:20216100].groupby(['timestamp', 'building_id', 'meter']).agg([sum])
nan = pd.concat([nan1, nan2, nan3, nan4])
nan['meter_reading', 'isNaN'] = np.isnan(nan)
nan.drop([('meter_reading', 'sum')], axis='columns', inplace=True)
nan.columns = nan.columns.droplevel(0)
nan.rename_axis(None, axis='columns')
nan.reset_index(inplace=True)
print('  Acquire existing timeseries')
existing_timeseries = X_train[['building_id', 'meter', 'meter_reading']].groupby(['building_id', 'meter']).agg(['count'])
existing_timeseries.columns = existing_timeseries.columns.droplevel(0)
existing_timeseries.rename_axis(None, axis='columns')
existing_timeseries.reset_index(inplace=True)
existing_timeseries.drop('count', axis='columns', inplace=True)
print('  Remove non-existing timeseries from NaN DataFrame')
nan['exists'] = False
for index, row in tqdm(existing_timeseries.iterrows(), total=existing_timeseries.shape[0]):
    nan.loc[(nan['building_id']==row['building_id']) & (nan['meter']==row['meter']), 'exists'] = True
nan = nan[nan['exists']==True]
nan.drop(['exists'], axis='columns', inplace=True)

print(' Export')
with open("nan.pickle", "wb") as fp:
    pickle.dump(nan, fp)

print("> Done !")
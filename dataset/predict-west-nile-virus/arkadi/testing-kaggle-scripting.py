import pandas as pd
import os

os.system("ls ../input")

train = pd.read_csv("../input/train.csv")
weather = pd.read_csv("../input/weather.csv")
spray = pd.read_csv("../input/spray.csv")

print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

print(train.head())

print(train[['Trap', 'Date', 'WnvPresent']].groupby(('Trap', 'Date')).sum())

test_cols = ['Id', 'Date', 'Species', 'Trap', 
    'Latitude', 'Longitude', 'NumMosquitos', 'WnvPresent']

# Only care for numeric values of weather
# average the two stations
weather_cols = ['Station', 'Date', 'Tmax', 'Tmin', 'Tavg', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'PrecipTotal', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed']
weather = weather[weather_cols].groupby('Date').mean()
print(weather.head())

def get_features(train, weather, spray):
    train_feats = train[['Id', 'Date']]
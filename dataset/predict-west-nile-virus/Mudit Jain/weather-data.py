import pandas as pd
import os

os.system("ls ../input")

#train = pd.read_csv("../input/train.csv")
#print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

#print(train.head())

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing

# Load dataset 
#train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('../input/test.csv')
#sample = pd.read_csv('../input/sampleSubmission.csv')
weather = pd.read_csv('../input/weather.csv')

# Split station 1 and 2 and join horizontally
weather_stn1 = weather[weather['Station']==1]
weather_stn2 = weather[weather['Station']==2]
weather_stn1 = weather_stn1.drop('Station', axis=1)
weather_stn2 = weather_stn2.drop('Station', axis=1)
weather = weather_stn1.merge(weather_stn2, on='Date')

print(weather.columns.tolist())
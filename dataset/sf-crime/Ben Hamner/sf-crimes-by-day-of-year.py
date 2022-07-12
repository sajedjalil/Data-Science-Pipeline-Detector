#Import and preparation of data
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import datetime as df
import time

z_train = zipfile.ZipFile('../input/train.csv.zip')
z_test  = zipfile.ZipFile('../input/test.csv.zip')
train = pd.read_csv(z_train.open('train.csv'), parse_dates=['Dates'])
test  = pd.read_csv(z_test.open('test.csv'),   parse_dates=['Dates'])

#Add day of the year format 02-22
train['DayOfYear'] = train['Dates'].map(lambda x: x.strftime("%m-%d"))
test['DayOfYear']  = test['Dates'].map( lambda x: x.strftime("%m-%d"))

train_days = train[["X", "DayOfYear"]].groupby(['DayOfYear']).count().rename(columns={"X": "TrainCount"})
test_days  = test[ ["X", "DayOfYear"]].groupby(['DayOfYear']).count().rename(columns={"X": "TestCount"})

days = train_days.merge(test_days, left_index=True, right_index=True)
days["TotalCount"] = days["TrainCount"] + days["TestCount"]

days.plot(figsize=(15,10)) 
plt.title("The two peaks per month pattern is entirely explained by splitting the data into train/test sets")
plt.ylabel('Number of crimes')
plt.xlabel('Day of year')
plt.grid(True)

plt.savefig('Distribution_of_Crimes_by_DayofYear.png')

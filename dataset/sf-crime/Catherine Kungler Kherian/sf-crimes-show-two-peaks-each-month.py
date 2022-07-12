#Import and preparation of data
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import datetime as df
import time

z = zipfile.ZipFile('../input/train.csv.zip')
train = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'])

#Add day of the year format 02-22
train['DayOfYear'] = train['Dates'].map(lambda x: x.strftime("%m-%d"))

df = train[['Category','DayOfYear']].groupby(['DayOfYear']).count()

df.plot(y='Category', label='Number of events', figsize=(15,10)) 
plt.title("Crimes occur with a regular pattern: two peaks per month")
plt.ylabel('Number of crimes')
plt.xlabel('Day of year')
plt.grid(True)

plt.savefig('Distribution_of_Crimes_by_DayofYear.png')

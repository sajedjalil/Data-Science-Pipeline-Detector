import pandas as pd
import zipfile
import matplotlib.pyplot as pl

z = zipfile.ZipFile('../input/train.csv.zip')
print(z.namelist())

train = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'])

# plot of the DayofWeek counts
print(train.DayOfWeek.value_counts())
train.DayOfWeek.value_counts().plot(kind='barh', title= 'Dayofweek counts', figsize=(7,7))
pl.savefig('Dayofweek_counts.png')


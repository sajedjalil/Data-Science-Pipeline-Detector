import pandas as pd
import zipfile
import matplotlib.pyplot as pl

z = zipfile.ZipFile('../input/train.csv.zip')
print(z.namelist())

train = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'])

train['Year'] = train['Dates'].map(lambda x: x.year)
train['Week'] = train['Dates'].map(lambda x: x.week)
train['Hour'] = train['Dates'].map(lambda x: x.hour)

print(train.head())

train.Category.value_counts().plot(kind='barh', figsize=(12,8))
pl.savefig('category_counts.png')

train.DayOfWeek.value_counts().plot(kind='barh', figsize=(10,10))
pl.savefig('Dayofweek_counts.png')

train.Resolution.value_counts().plot(kind='barh', figsize=(12,10))
pl.savefig('Resolution_counts.png')

train['event']=1
weekly_events = train[['Week','Year','event']].groupby(['Year','Week']).count().reset_index()
print(weekly_events)
weekly_events_years = weekly_events.pivot(index='Week', columns='Year', values='event').fillna(method='ffill')
#%matplotlib inline
ax = weekly_events_years.interpolate().plot(title='number of cases every 2 weeks', figsize=(10,6))
pl.savefig('events_every_two_weeks.png')


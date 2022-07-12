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

train.PdDistrict.value_counts().plot(kind='bar', figsize=(8,10))
pl.savefig('district_counts.png')


train['event']=1
weekly_events = train[['Week','Year','event']].groupby(['Year','Week']).count().reset_index()
weekly_events_years = weekly_events.pivot(index='Week', columns='Year', values='event').fillna(method='ffill')
#%matplotlib inline
ax = weekly_events_years.interpolate().plot(title='number of cases every 2 weeks', figsize=(10,6))
pl.savefig('events_every_two_weeks.png')


hourly_events = train[['Hour','event']].groupby(['Hour']).count().reset_index()
hourly_events.plot(kind='bar', figsize=(6, 6))
pl.savefig('hourly_events.png')


hourly_district_events = train[['PdDistrict','Hour','event']].groupby(['PdDistrict','Hour']).count().reset_index()
hourly_district_events_pivot = hourly_district_events.pivot(index='Hour', columns='PdDistrict', values='event').fillna(method='ffill')
hourly_district_events_pivot.interpolate().plot(title='number of cases hourly by district', figsize=(10,6))
pl.savefig('hourly_events_by_district.png')

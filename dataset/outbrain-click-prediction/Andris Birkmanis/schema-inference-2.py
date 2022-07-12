import pandas as pd

# find out if events are a subset of views
events=pd.read_csv('../input/events.csv',usecols=['uuid','timestamp'])
views=pd.read_csv('../input/page_views_sample.csv',usecols=['uuid','timestamp'])
x=pd.merge(events,views,on=['uuid','timestamp'])
x.to_csv('events_in_views.csv')
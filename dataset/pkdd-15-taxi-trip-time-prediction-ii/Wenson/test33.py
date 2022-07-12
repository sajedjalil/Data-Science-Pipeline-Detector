
import json
import zipfile
import numpy as np
import pandas as pd
# from __future__ import division

### Control the number of trips read for training 
### Control the number of closest trips used to calculate trip duration
N_read = 50000
N_trips = 200

### Get Haversine distance
def get_dist(lonlat1, lonlat2):
  lon_diff = np.abs(lonlat1[0]-lonlat2[0])*np.pi/360.0
  lat_diff = np.abs(lonlat1[1]-lonlat2[1])*np.pi/360.0
  a = np.sin(lat_diff)**2 + np.cos(lonlat1[1]*np.pi/180.0) * np.cos(lonlat2[1]*np.pi/180.0) * np.sin(lon_diff)**2  
  d = 2*6371*np.arctan2(np.sqrt(a), np.sqrt(1-a))
  return(d)

# read test
zf = zipfile.ZipFile('../input/test.csv.zip')
test = pd.read_csv(zf.open('test.csv'), usecols=['TRIP_ID', 'POLYLINE'])
test['POLYLINE'] = test['POLYLINE'].apply(json.loads)
test['snapshots'] = test['POLYLINE'].apply(len)
test['lonlat'] = test['POLYLINE'].apply(lambda x: x[0])
test.drop('POLYLINE', axis=1, inplace=True)

# read train
zf = zipfile.ZipFile('../input/train.csv.zip')
train = pd.read_csv(zf.open('train.csv'), usecols=['POLYLINE'], nrows=N_read)
train['POLYLINE'] = train['POLYLINE'].apply(json.loads)
train['snapshots'] = train['POLYLINE'].apply(len)
train = train[train.snapshots>25]
train['lonlat'] = train['POLYLINE'].apply(lambda x: x[0])
train.drop('POLYLINE', axis=1, inplace=True)

test['TRAVEL_TIME'] = 0
for row, ll in enumerate(test['lonlat']):	
   	### Weighted mean of trip duration
   	### Bound below by 10 meters since we use 1/distance^2 as weight
  	### Treat 10% of longest lasting trips as outliers  	
	d = train['lonlat'].apply(lambda x: get_dist(x, ll))
	i = np.argpartition(d, N_trips)[0:N_trips]
	w = np.maximum(d.iloc[i], 0.01)
	s = train.iloc[i]['snapshots']
	j = np.argpartition(s, int(N_trips*.95))[0:int(N_trips*.95)]
	test.loc[row, 'TRAVEL_TIME'] = 15*np.maximum(test.loc[row, 'snapshots'], np.average(s.iloc[j], weights=1/w.iloc[j]**2))

test['TRAVEL_TIME'] = test['TRAVEL_TIME'].astype(int)
test[['TRIP_ID', 'TRAVEL_TIME']].to_csv('submission.csv', index=False)

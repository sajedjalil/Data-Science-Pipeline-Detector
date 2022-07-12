import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from collections import Counter

def getFreq(df):
   freq = Counter(df['Agencia_ID'])
   df1 = pd.DataFrame(list(freq.keys()), columns=['Agencia_ID'])
   df2 = pd.DataFrame(list(freq.values()), columns=['count'])
   freq_df = pd.concat([df1, df2], axis=1)
   return freq_df

def getSpatialFeatures(df):
   geolocator = Nominatim()
   towns = list(df['Town'])
   locs = []
   for town in towns:
      print (town)
      loc = geolocator.geocode(str(town))
      if loc is not None:
         print (loc.raw)
         locs.append(loc.raw)
      else:
         print ('fail')
         locs.append({})
   df_loc = pd.DataFrame(locs)
   return df_loc

def prepareResults(df_res, df_loc):
    df_res = pd.concat([df_res, df_loc], axis=1)
    #scale the column to the range from 8 to 88
    df_res['count'] -= df_res['count'].min()
    df_res['count'] /= df_res['count'].max()
    df_res['count'] = df_res['count']*80+8
    df_res['count'] = df_res['count'].astype(int)
    #drop rows with no location information
    df_res = df_res.dropna()
    #lat and lon to float format
    df_res['lat'] = df_res['lat'].astype(float)
    df_res['lon'] = df_res['lon'].astype(float)
    return df_res

def drawMap(df):
   #from http://chrisalbon.com/python/matplotlib_plot_points_on_map.html
   fig = plt.figure(figsize=(20,10))
   m = Basemap(projection='gall', 
              # with low resolution,
              resolution = 'l', 
              # And threshold 100000
              area_thresh = 100000.0,
              #llcrnrlon=-136.25, llcrnrlat=56.0,
              #urcrnrlon= -180.0, urcrnrlat=90.0,
              # Centered at 0,0 (i.e null island)
              lat_0=20.0, lon_0=0)

   # Draw the coastlines on the map
   m.drawcoastlines()
   # Draw country borders on the map
   m.drawcountries()
   # Fill the land with grey
   m.fillcontinents(color = '#888888')
   # Draw the map boundaries
   m.drawmapboundary(fill_color='#f4f4f4')
   x,y = m(df['lon'].values, df['lat'].values)
   z = df['count'].values
   print (z)
   # Plot them using round markers of size 6
   #m.plot(x, y, 'ro', markersize=3)
   m.scatter(x, y, s=z, c='r', marker='o', zorder=2)
   # Show the map
   plt.show()
   plt.savefig('test.png')
    
df = pd.read_csv('../input/train.csv', usecols=['Agencia_ID'])
dfl = pd.read_csv('../input/town_state.csv')
#calculate the frequencies of the locations in tes
freq_df = getFreq(df)
df_res = pd.merge(freq_df, dfl, how='left', on='Agencia_ID')
#extract spatial features from the names of the towns
df_loc = getSpatialFeatures(df_res)
#df_loc.to_csv('loc_features.csv', index=False, encoding='utf-8')
#df_loc = pd.read_csv('loc_features.csv')
df_res = prepareResults(df_res, df_loc)
drawMap(df_res)

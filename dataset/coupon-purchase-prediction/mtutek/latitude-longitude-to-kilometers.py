import pandas as pd
import numpy as np

# Distance by the Haversine formula
# https://en.wikipedia.org/wiki/Haversine_formula
def dist_from_coordinates(lat1, lon1, lat2, lon2):
    R = 6371 # Earth's radius in KM

    # convert to radians
    d_lat = np.radians(lat2-lat1)
    d_lon = np.radians(lon2-lon1)

    r_lat1 = np.radians(lat1)
    r_lat2 = np.radians(lat2)

    # argument under the root
    a = np.sin(d_lat/2.) **2 + np.cos(r_lat1) * np.cos(r_lat2) * np.sin(d_lon/2.)**2

    haversine = 2 * R * np.arcsin(np.sqrt(a))

    return haversine

# error because the file is not available to scripts, you can download it from the data page
##pref_locations = pd.read_csv('../input/prefecture_locations.csv')

# to test
##lats = pref_locations['LATITUDE']
##longs = pref_locations['LONGITUDE']

# point 1 - Sapporo
lat1 = 43.063968 #lats.values[0]
lon1 = 141.347899 #longs.values[0]

# point 2 - Aomori
lat2 = 40.824623 #lats.values[1]
lon2 = 140.740593 #longs.values[1]

# 254.016997815 - checks out on google maps
# 0.5% error possible due to earth being a spheroid

print(dist_from_coordinates(lat1, lon1, lat2, lon2), 'km')

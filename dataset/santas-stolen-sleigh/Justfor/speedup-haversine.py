import pandas as pd
import numpy as np
#from haversine import haversine

R = 6371.0  # radius of the earth in km
    
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = R * c
    return km

def haversine_est(lon1, lat1, lon2, lat2):
    """
    Estimation with equirectangular distance approximation. 
    Since the distance is relatively small, you can use the equirectangular distance approximation. 
    This approximation is faster than using the Haversine formula. 
    So, to get the distance from your reference point (lat1/lon1) to the point your are testing (lat2/lon2),
    use the formula below. 
    Important Note: you need to convert all lat/lon points to radians:
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    x = (lon2 - lon1) * np.cos( 0.5*(lat2+lat1) )
    y = lat2 - lat1
    km = R * np.sqrt( x*x + y*y )
    return km

# Example benchmark
lon1, lon2, lat1, lat2 = np.random.randn(4, 1000000)
df = pd.DataFrame(data={'lon1':lon1,'lon2':lon2,'lat1':lat1,'lat2':lat2})

import time

#print("Standard haversine()")
#t0 = time.time()
#km1 = haversine(df['lon1'],df['lat1'],df['lon2'],df['lat2'])
#t1 = time.time()
#print (t1-t0)

print("Numpy haversine()")
t0 = time.time()
km2 = haversine_np(df['lon1'],df['lat1'],df['lon2'],df['lat2'])
t1 = time.time()
print (t1-t0)

print("haversine() estimated")
t0 = time.time()
km3 = haversine_est(lon1, lat1, lon2, lat2)
t1 = time.time()
print (t1-t0)

print("Timeit")
import timeit
t1 = timeit.repeat("haversine_np(df['lon1'],df['lat1'],df['lon2'],df['lat2'])", 
"from __main__ import haversine_np; from __main__ import df;",
number=100)
print("Numpy haversine()")
print(t1)
                  
t2=timeit.repeat("haversine_est(df['lon1'],df['lat1'],df['lon2'],df['lat2'])", 
"from __main__ import haversine_est; from __main__ import df",number=100)
print("haversine() estimated")
print(t2)


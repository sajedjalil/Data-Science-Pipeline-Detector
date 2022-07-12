import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x=pd.read_csv('../input/events.csv')
gd=x.groupby('device_id')

# Number of people to show
i = 2000

# Threshold of number of data points for person to be included (discluding 0,0s)
thresh = 10

plt.figure(figsize=(25,20)) 
for g in gd:
    g = g[1]
    # Limit to china only
    g = g.loc[g.latitude > 18]
    g = g.loc[g.longitude > 80]
    if g.shape[0] > thresh:
        lon = g['longitude']
        lat = g['latitude']
        plt.plot(lon, lat, '-o', color=np.random.rand(3).tolist())
        i -= 1
    if i < 0:
        break
    
plt.savefig('gps_map.png')
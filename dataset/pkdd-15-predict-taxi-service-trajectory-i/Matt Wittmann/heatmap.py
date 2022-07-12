import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile

bins = 1000
lat_min, lat_max = 41.04961, 41.24961
lon_min, lon_max = -8.71099, -8.51099

with ZipFile('../input/train.csv.zip') as zf:
    
    data = pd.read_csv(zf.open('train.csv'),
                       chunksize=10000,
                       usecols=['POLYLINE'],
                       converters={'POLYLINE': lambda x: json.loads(x)})
    
    # process data in chunks to avoid using too much memory
    z = np.zeros((bins, bins))
    
    for chunk in data:

        latlon = np.array([(lat, lon) 
                           for path in chunk.POLYLINE
                           for lon, lat in path if len(path) > 0])

        z += np.histogram2d(*latlon.T, bins=bins, 
                            range=[[lat_min, lat_max],
                                   [lon_min, lon_max]])[0]
        
log_density = np.log(1+z)

plt.imshow(log_density[::-1,:], # flip vertically
           extent=[lat_min, lat_max, lon_min, lon_max])


plt.savefig('heatmap.png')
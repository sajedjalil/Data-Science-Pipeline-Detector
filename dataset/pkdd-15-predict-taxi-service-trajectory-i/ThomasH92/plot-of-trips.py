import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Longitude and latitude coordinates of Porto
lat_mid = 41.1496100
lon_mid = -8.6109900

# Load the data
zf = zipfile.ZipFile('../input/train.csv.zip')
df = pd.read_csv(zf.open('train.csv'), 
                 sep = ",",
                 chunksize = 1000,
                 iterator = True,
                 usecols = ['POLYLINE'],
                 converters={'POLYLINE': lambda x: json.loads(x)})
  
nrbins = 2000
hist = np.zeros((nrbins,nrbins))

for data in df:
  # Get just the longitude and latitude coordinates for each trip
  latlong = np.array([ coord for coords in data['POLYLINE'] for coord in coords if len(coords) > 0])

  # Compute the histogram with the longitude and latitude data as a source
  hist_new, _, _  = np.histogram2d(x = latlong[:,1], y = latlong[:,0], bins = nrbins, 
                                   range = [[lat_mid - 0.1, lat_mid + 0.1], [lon_mid - 0.1, lon_mid + 0.1]])
                                   
  # Add the new counts to the previous counts
  hist = hist + hist_new
  

# We consider the counts on a logarithmic scale
img = np.log(hist[::-1,:] + 1)

# Plot the counts
plt.figure()
ax = plt.subplot(1,1,1)
plt.imshow(img)
plt.axis('off')
      
plt.savefig('trips_density.png')
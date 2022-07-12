"""This is an attempt to visualize both the temporal and spatial distributions of crimes"""

__author__='Lian L'
__version__='0.3'
__date__='23/07/2015'


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('ggplot')
train_data=pd.read_csv('../input/train.csv')

# We want to create a scatterplot of crime occurences for the whole city
# Borrowing the map and information from Ben's script
SF_map= np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")
print(SF_map)
# Supplied map bounding box:
#    ll.lon     ll.lat   ur.lon     ur.lat
#    -122.52469 37.69862 -122.33663 37.82986
lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
asp = SF_map.shape[0] * 1.0 / SF_map.shape[1]
fig = plt.figure(figsize=(16,16))
plt.imshow(SF_map,cmap='gray',extent=lon_lat_box,aspect=asp)#previous aspect=1/asp
ax=plt.gca()
# Discard some entries with erratic position coordinates
train_data[train_data['Y']<40].plot(x='X',y='Y',ax=ax,kind='scatter',marker='o',s=2,color='green',alpha=0.01)
ax.set_axis_off()
plt.savefig('TotalCrimeonMap.png')
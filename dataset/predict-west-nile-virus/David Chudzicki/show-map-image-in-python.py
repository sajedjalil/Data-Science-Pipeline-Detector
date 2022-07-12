# This shows how to read the text representing a map of Chicago in numpy, and put it on a plot in matplotlib.
# This example doesn't make it easy for you to put other data in lat/lon coordinates on the plot.
# Hopefully someone else can add an example showing how to do that? You'll need to know the bounding box of this map:
#    ll.lat ll.lon ur.lat ur.lon
#    41.6    -88   42.1  -87.5

import numpy as np
import matplotlib.pyplot as plt

mapdata = np.loadtxt("../input/mapdata_copyright_openstreetmap_contributors.txt")
plt.imshow(mapdata, cmap = plt.get_cmap('gray'))
plt.savefig('map.png')
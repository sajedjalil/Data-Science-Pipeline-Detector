# This shows how to read the text representing a map of Chicago in numpy, and put it on a plot in matplotlib.
# This example doesn't make it easy for you to put other data in lat/lon coordinates on the plot.
# Hopefully someone else can add an example showing how to do that? You'll need to know the bounding box of this map:
#    ll.lon     ll.lat   ur.lon     ur.lat
#    -122.52469 37.69862 -122.33663 37.82986

import numpy as np
import matplotlib.pyplot as plt

mapdata = np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")
plt.imshow(mapdata, cmap = plt.get_cmap('gray'))
plt.savefig('map.png')

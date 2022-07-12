# This shows how to read the text representing a map of Chicago in numpy, and put it on a plot in matplotlib.
# This example also rescales the image data to the GPS co-ordinates of the bounding box and overlays some random points.

import numpy as np
import matplotlib.pyplot as plt

origin = [41.6, -88.0]              # lat/long of origin (lower left corner)
upperRight = [42.1, -87.5]          # lat/long of upper right corner

mapdata = np.loadtxt("../input/mapdata_copyright_openstreetmap_contributors.txt")


# generate some data to overlay
numPoints = 50
lats = (upperRight[0] - origin[0]) * np.random.random_sample(numPoints) + origin[0]
longs = (upperRight[1] - origin[1]) * np.random.random_sample(numPoints) + origin[1]

intersection = [41.909614, -87.746134]  # co-ordinates of intersection of IL64 / IL50 according to Google Earth


# generate plot
plt.imshow(mapdata, cmap=plt.get_cmap('gray'), extent=[origin[1], upperRight[1], origin[0], upperRight[0]])
plt.scatter(x=longs, y=lats, c='r', s=20)
plt.scatter(x=intersection[1], y=intersection[0], c='b', s=60, marker='s')

#plt.show()
plt.savefig('map.png')
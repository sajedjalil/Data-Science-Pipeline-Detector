import numpy as np
import matplotlib.pyplot as plt

mapdata = np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")
plt.imshow(mapdata, cmap = plt.get_cmap('gray'))
plt.savefig('map.png')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
###import data
data = pd.read_json("../input/train.json")

# Set up plot
data_sample = data.sample(n=1000)
fig, ax = plt.subplots(figsize=(6,6))
# Mercator of World
lon_min, lon_max = -74.289551, -73.700165
lat_min, lat_max = 40.477398, 40.91758

m1 = Basemap(projection='merc',
             llcrnrlat=lat_min,
             urcrnrlat=lat_max,
             llcrnrlon=lon_min,
             urcrnrlon=lon_max,
             lat_ts=40.7,
             resolution='i')

m1.drawmapboundary(fill_color='#46bcec')
m1.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m1.drawcountries(linewidth=0.1, color="#000000")

# Plot the data
mxy = m1(data_sample["longitude"].tolist(), data_sample["latitude"].tolist())
m1.scatter(mxy[0], mxy[1], s=3, c="#1292db", lw=0, alpha=1, zorder=5)

plt.show()
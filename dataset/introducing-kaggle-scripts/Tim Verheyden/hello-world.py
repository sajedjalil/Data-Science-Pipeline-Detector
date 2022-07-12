from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

map = Basemap(projection='ortho', lat_0=50, lon_0=-100,resolution='l', area_thresh=1000.0)
map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color='tan')
map.drawmapboundary()
plt.text(10,10, 'HELLO', horizontalalignment='center', verticalalignment='center', fontsize=80, color='blue')
plt.savefig('plot.png')
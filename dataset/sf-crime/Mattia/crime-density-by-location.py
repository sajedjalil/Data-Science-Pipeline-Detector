import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as pl
import seaborn as sns

# Supplied map bounding box:
#    ll.lon     ll.lat   ur.lon     ur.lat
#    -122.52469 37.69862 -122.33663 37.82986
mapdata = np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")
asp = mapdata.shape[0] * 1.0 / mapdata.shape[1]

lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]

z = zipfile.ZipFile('../input/train.csv.zip')
train = pd.read_csv(z.open('train.csv'))

#Get rid of the bad lat/longs
train['Xok'] = train[train.X<-121].X
train['Yok'] = train[train.Y<40].Y
train = train.dropna()
trainP = train[train.Category == 'PROSTITUTION'] #Grab the prostitution crimes
train = train[1:300000] #Can't use all the data and complete within 600 sec :(

#Seaborn FacetGrid, split by crime Category
g= sns.FacetGrid(train, col="Category", col_wrap=6, size=5, aspect=1/asp)

#Show the background map
for ax in g.axes:
    ax.imshow(mapdata, cmap=pl.get_cmap('gray'), 
              extent=lon_lat_box, 
              aspect=asp)
#Kernel Density Estimate plot
g.map(sns.kdeplot, "Xok", "Yok", clip=clipsize)

pl.savefig('category_density_plot.png')

#Do a larger plot with prostitution only
pl.figure(figsize=(20,20*asp))
ax = sns.kdeplot(trainP.Xok, trainP.Yok, clip=clipsize, aspect=1/asp)
ax.imshow(mapdata, cmap=pl.get_cmap('gray'), 
              extent=lon_lat_box, 
              aspect=asp)
pl.savefig('prostitution_density_plot.png')
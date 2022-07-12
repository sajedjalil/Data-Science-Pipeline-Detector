import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as pl
import seaborn as sns

# Run this script on Kaggle.
# It is forked from https://www.kaggle.com/dbennett/sf-crime/test-map/code

# Supplied map bounding box:
#    ll.lon     ll.lat   ur.lon     ur.lat
#    -122.52469 37.69862 -122.33663 37.82986
mapdata = np.loadtxt('../input/sf_map_copyright_openstreetmap_contributors.txt')
asp = mapdata.shape[0] * 1.0 / mapdata.shape[1]

# Map range
lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366], [37.699, 37.8299]]

z = zipfile.ZipFile('../input/train.csv.zip')
train = pd.read_csv(z.open('train.csv'))

#Get rid of the bad lat/longs
train['Longitude'] = train[train.X < -121].X
train['Latitude'] = train[train.Y < 40].Y
train = train.dropna()

#Grab the drug crimes
drug_data = train[train.Category == 'DRUG/NARCOTIC']
# Do a larger plot with prostitution only
pl.figure(figsize=(20, 20 * asp))
ax = sns.kdeplot(drug_data.Longitude,
                 drug_data.Latitude,
                 clip=clipsize,
                 aspect=1 / asp)
ax.imshow(mapdata, cmap=pl.get_cmap('gray'), extent=lon_lat_box, aspect=asp)
pl.savefig('drug_density_plot.png')

#Grab the drug crimes
vehicle_theft_data = train[train.Category == 'VEHICLE THEFT']
# Do a larger plot with prostitution only
pl.figure(figsize=(20, 20 * asp))
ax = sns.kdeplot(vehicle_theft_data.Longitude,
                 vehicle_theft_data.Latitude,
                 clip=clipsize,
                 aspect=1 / asp)
ax.imshow(mapdata, cmap=pl.get_cmap('gray'), extent=lon_lat_box, aspect=asp)
pl.savefig('vehicle_theft_density_plot.png')

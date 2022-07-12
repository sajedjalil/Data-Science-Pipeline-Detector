import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mapdata = np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")
plt.imshow(mapdata,cmap=plt.get_cmap('Blues_r'))
plt.axis('off')

z = zipfile.ZipFile('../input/train.csv.zip')
train = pd.read_csv(z.open('train.csv'))


#print (mapdata.shape)
#define the aspect ratio of the mapdata
asp = mapdata.shape[1]/mapdata.shape[0]

# train[train.X<-121] returns part of DATAFRAME that meets the criteria as a new DATAFRAME!
# train[train.X<-121].X returns the X column of the dataframe.
train['Xok'] = train[train.X<-121].X
train['Yok'] = train[train.Y<40].Y
train = train.dropna()
trainP = train[train.Category == 'PROSTITUTION'] #Grab the prostitution crimes to trainP (datafr)
train = train.iloc[0:30000]

print (train.info())
lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]


'''
g = sns.FacetGrid(train,col='Category', size = 5, col_wrap = 6, aspect = asp)
# Plot a facet-grid based on the data 'Category' in train 



g.axes[1].imshow(mapdata, cmap=plt.get_cmap('gray'), 
              extent=lon_lat_box, 
              aspect=asp)
              
'''


#g.map(sns.kdeplot, "Xok", "Yok", clip=clipsize)



'''
for ax in g.axes:
    ax.imshow(mapdata, cmap=plt.get_cmap('gray'), 
              extent=lon_lat_box, 
              aspect=asp)
#Kernel Density Estimate plot
g.map(sns.kdeplot, "Xok", "Yok", clip=clipsize)
'''

plt.figure(figsize=(20,20/asp))
ax = sns.kdeplot(trainP.Xok, trainP.Yok,clip=clipsize,aspect=asp)
ax.imshow(mapdata,cmap=plt.get_cmap('gray'), 
              extent=lon_lat_box, 
              aspect=asp)
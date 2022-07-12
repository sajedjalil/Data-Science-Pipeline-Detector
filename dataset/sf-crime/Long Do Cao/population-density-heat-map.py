import shapefile as shp #download at https://github.com/GeospatialPython/pyshp
import urllib
import zipfile
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection



### GET POPULATION DENSITY FILE
##download file
#make sure your file is 467kb
url = "https://data.sfgov.org/download/ea9w-4zvc/SHAPEFILE/SanFranciscoPopulation.zip"
destdir = "../data/"
name = "SanFranciscoPopulation"
destfile = destdir+name+".zip"

if not os.path.isfile(destfile) :
    urllib.urlretrieve(url, destfile)
    zfile = zipfile.ZipFile(destfile)
    zfile.extractall(destdir)

##read shapefile
sf = shp.Reader(destdir+name)









###PLOT
##get the shapes
patches=[]
fig, ax = plt.subplots()
for shape in sf.shapeRecords(): #loop over polygon
    x = [i[0] for i in shape.shape.points[:]] #get all x coord of one polygon
    y = [i[1] for i in shape.shape.points[:]] #get all y coord of one polygon
    plt.plot(x,y,"k") #draw contours in black
    polygon = Polygon(np.array([x,y]).T, closed=True) #get one single polygon
    patches.append(polygon) #add it to a final array


##get the population density as color code
pop_density = [rec[3] for rec in sf.records()]
pop_density = np.array(map(float,pop_density)) #convert to float
colors = pop_density/max(pop_density) #normalize color

##now plot    
p = PatchCollection(patches, cmap="Blues")
p.set_array(colors)
ax.add_collection(p)
plt.show()

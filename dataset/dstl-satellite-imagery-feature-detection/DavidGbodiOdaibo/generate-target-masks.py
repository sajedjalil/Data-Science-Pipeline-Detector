# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import os
import pandas
import numpy as np 
import scipy.misc
from shapely import wkt
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
from shapely import wkt
from shapely import affinity
import shapely
from PIL import Image,   ImageDraw
import pylab
from scipy import misc


datasetRoot = '../input/'

########################################################################################
# read the training data from train_wkt_v4.csv
df = pd.read_csv( datasetRoot + 'train_wkt_v4.csv')
 
# grid size will also be needed later..
gs = pd.read_csv(  datasetRoot + 'grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
 

# imageIds in a DataFrame
allImageIds = gs.ImageId.unique()
trainImageIds = df.ImageId.unique()

##################################################################

polygons_raw = pd.read_csv(datasetRoot + 'train_wkt_v4.csv')
grid_sizes = pd.read_csv(datasetRoot + 'grid_sizes.csv')
cols = grid_sizes.columns.tolist()
cols[0]='ImageId'
grid_sizes.columns = cols

count = 0
for i,img_id in enumerate( allImageIds):

    if img_id in trainImageIds:
        image = tiff.imread(datasetRoot + 'three_band/'+ img_id+'.tif')
        for iclass in range(9, 10):
            i_grid_size = grid_sizes[grid_sizes.ImageId == img_id]
            x_max = i_grid_size.Xmax.values[0]
            y_min = i_grid_size.Ymin.values[0]

            #Get just a single class of training polygons for this image
            class_2 = polygons_raw[(polygons_raw.ImageId == img_id) & (polygons_raw.ClassType==iclass)]

            #WKT to shapely object
            polygons = wkt.loads(class_2.MultipolygonWKT.values[0])
            W = float(int(image.shape[1] / 10))
            H = float(int(image.shape[2] / 10))


            #Transform the polygons 
            W_ = W * (W/(W+1))
            H_ = H * (H/(H+1))

            x_scaler = W_ / x_max
            y_scaler = (H_ / y_min ) #* -1.0

            polygons = shapely.affinity.scale(polygons, xfact = x_scaler, yfact= y_scaler, origin=(0,0,0))
            img = Image.new('L', (int(W), int(H)), 0)
            if len(polygons) > 0:
                for pol in polygons:
                    draw = ImageDraw.Draw(img)
                    #print(list(pol.exterior.coords))
                    draw.polygon(list(pol.exterior.coords), outline=1, fill=1)
                    for inte in pol.interiors:
                        draw.polygon(list(inte.coords), outline=1, fill=0)
                    del draw

            
            a = np.zeros((350,350))
            a[0:np.array(img).shape[0],0:np.array(img).shape[1]] = np.array(img)
 
            cv2.imwrite(img_id+'class_'+str(iclass)+'.png',a*255)
        count = count + 1
        if count > 18:
            break
    
 
       
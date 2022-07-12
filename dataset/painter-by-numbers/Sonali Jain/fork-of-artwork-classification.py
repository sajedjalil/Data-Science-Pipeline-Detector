# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage import morphology as mph
from skimage.data import imread
from skimage.data import load
from skimage import transform as trf
from skimage import measure as meas
import skimage.exposure as expos
import matplotlib.pyplot as plt
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
print(check_output(["ls", "../input/train_2"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train_info=pd.read_csv("../input/train_info.csv", encoding="utf_8")
print("Original record size", train_info.shape)
recordClean=train_info.loc[-train_info.isnull().any(axis=1), :]
recordNaN=train_info.loc[train_info.isnull().any(axis=1), :]
print("Clean record size", recordClean.shape)
print("Records with NaN values", recordNaN.shape)

mydata=[]
#for i in recordClean["filename"]:
for i in recordClean["filename"].iloc[range(1000)]:
    pos_file=os.path.join("../input/train_"+i[0], i)
    #print(pos_file)
    if os.path.isfile(pos_file):
        print("accessing!", pos_file)
        mydata.append(imread(pos_file))

print("Total valid data record found", len(mydata))
for m in range(len(mydata)):
    print("For each record, we found", mydata[m].shape, "data points")

# DAMN, we need to rescale the images to a uniform size? they are of different sizes..
# Maybe not, we might just have to do image analyses on each image and use those
# universal metrics (e.g. color histogram, intensity distribution, spatial frequency)
# , plus some already-known metrics, such as "genre" and "year", to do 
# classification/cross-check?

# 1. Each artist might have a particular preference on color combinations
#    Probably unique, thus a distrubtion of color is useful

# 2. The delicacy of an image might be a useful metric for artist style. Some styles are just
#    doodles, but some are very substantial-- their pictures are like photos
#    Spatial frequency, or average interspacing of different coloring pixels might be useful

# 3. Similar to metric 1, the intensity (how dark) of an image might or might not be a useful
#    metric. If not, we should just normalize the intensity?

fig=plt.figure(figsize=(5,5))
fig2=plt.figure(figsize=(5,5))
moment_data=np.ndarray((9,16,3))
for j in range(1,10):
    combhist=[]
    for i in range(3):
        combhist.append(expos.histogram(mydata[j-1][:,:,i]))
        m=meas.moments(mydata[j-1][:,:,i])
        m.shape=16
        moment_data[j-1,:,i]=m
    ax=fig.add_subplot(3,3,j)
    for i in range(3):
        ax.plot(combhist[i][1],combhist[i][0])


for i in range(3):
    for j in range(16):
        ax2=fig2.add_subplot(4,4,j+1)
        ax2.plot(range(9), moment_data[:,j,i])
        
#fig2.savefig(fname)
    
fig.savefig("all.jpg")
fig2.savefig("moment.jpg")

# OK, the color histogram metric is useful... 
# the detailed feature (high order moment) also seem to be helpful as the values are
# different for each case

# In metric 1, we've got 255*3 features
# In metric 2, we've got 16*3 features  (more if we go to higher moments)

# We start the test with metric 1 only
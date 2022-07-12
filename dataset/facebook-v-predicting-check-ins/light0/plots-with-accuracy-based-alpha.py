# -*- coding: utf-8 -*-
"""
Created on Fri Jun 03 21:00:28 2016

@author: hardy_000
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


def loadtraindata():
    z= pd.read_csv("../input/train.csv")
    return z
    

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # By Jake VanderPlas
    # License: BSD-style
    
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
    


def getSamplePlaces(numberOfPlaces=1):
        
    return train[train.place_id.isin((train.place_id).unique()[1:numberOfPlaces+1])]
def getSamplePlace(numberOfPlaces=1):
        
    return train[train.place_id==(train.place_id).unique()[numberOfPlaces]]

def normalize(series):
    return (series-min(series))/(max(series)-min(series))   
    
train=loadtraindata()
sample=getSamplePlaces(30)
norm = mpl.colors.Normalize(vmin=min(sample.place_id),vmax=max(sample.place_id))
cp=plt.cm.get_cmap('jet')
cp=discrete_cmap(30,cp)
cols=cp(norm(sample['place_id']))
cols[:,3]=normalize(sample['accuracy'])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('time')
ax.scatter(sample.x,sample.y,sample.time,c=cols,marker='o',depthshade=False,lw = 0)
plt.savefig('output.png')

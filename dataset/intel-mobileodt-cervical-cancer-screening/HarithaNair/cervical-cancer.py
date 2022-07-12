import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread, imshow
import matplotlib.cm as cm
import cv2
"""
import plotly.offline as py
import plotly.garph_objs as go
import plotly.tools as tls
"""

from glob import glob
basepath = '../input/train/'

all_images=[]

for path in sorted(glob(basepath+"*")):
    typei=path.split("/")[-1]
    img= sorted(glob(basepath+typei+"/*"))
    all_images=all_images+img

all_images=pd.DataFrame({'imagepath':all_images})
all_images['filetype']=all_images.apply(lambda row: row.imagepath.split(".")[-1], axis=1)
all_images['type']=all_images.apply(lambda row: row.imagepath.split("/")[-2], axis=1)
print(all_images.shape)
"""
type_agg = (all_images.groupby(['type', 'filetype']).agg('count'))/1481
print(type_agg)
"""
fig = plt.figure(figsize=(12,8))

i=1
for t in all_images['type'].unique():
    ax=fig.add_subplot(1,3,i)
    i=i+1
    f=all_images[all_images['type']==t]['imagepath'].values[0]
    img=plt.imread(f)
    plt.title(t)
    plt.imshow(img, cmap=cm.binary)
    plt.savefig('output1.png')


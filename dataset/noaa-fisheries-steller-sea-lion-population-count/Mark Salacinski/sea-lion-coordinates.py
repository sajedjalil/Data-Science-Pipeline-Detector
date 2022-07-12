import sys
import os

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import glob

import cv2
import shapely
import shapely.geometry
from shapely.geometry import Polygon

import matplotlib.pyplot as plt


SOURCEDIR = os.path.join('..', 'input')

# ================ Meta ====================
__description__ = 'Sea Lion Prognostication Engine'
__version__ = '0.1.0'
__license__ = 'MIT'
__author__ = 'Gavin Crooks (@threeplusone)'
__status__ = "Prototype"
__copyright__ = "Copyright 2017"


class SeaLionData(object):
    
    def __init__(self, sourcedir=SOURCEDIR):
        
        self.class_names = (
            'adult_males',
            'subadult_males',
            'adult_females',
            'juveniles',
            'pups')
        
        self.class_colors = (
            (255,0,0),          # red
            (250,10,250),       # magenta
            (84,42,0),          # brown 
            (30,60,180),        # blue
            (35,180,20),        # green
            )

        self._trainshort_nb = 11 
        
        self.train_nb = 947
        
        self.test_nb = 18636
       
        self.source_paths = {
            'sample'     : os.path.join(sourcedir, 'sample_submission.csv'),
            'counts'     : os.path.join(sourcedir, 'Train', 'train.csv'),
            'train'      : os.path.join(sourcedir, 'Train', '{iid}.jpg'),
            'dotted'     : os.path.join(sourcedir, 'TrainDotted', '{iid}.jpg'),
            'test'       : os.path.join(sourcedir, 'Test', '{iid}.jpg'),   
            }
        
        # From MismatchedTrainImages.txt
        self.bad_train_ids = (
            3, 7, 9, 21, 30, 34, 71, 81, 89, 97, 151, 184, 215, 234, 242, 
            268, 290, 311, 331, 344, 380, 384, 406, 421, 469, 475, 490, 499, 
            507, 530, 531, 605, 607, 614, 621, 638, 644, 687, 712, 721, 767, 
            779, 781, 794, 800, 811, 839, 840, 869, 882, 901, 903, 905, 909, 
            913, 927, 946)

    def class_from_color(self, rgb) :
        MAX_DIFF = 80
        colors = np.array(self.class_colors) 
        diff = (abs(colors - rgb)).sum(axis=-1)
        cls = np.argmin(diff)
        
        if diff[cls]>MAX_DIFF: 
            #print(cls, rgb, self.class_colors[cls], diff[cls])
            return None
        
        return cls
        
    @property
    def trainshort_ids(self):
        return (0,1,2,4,5,6,8,10) #range(41,51) 
        
    @property 
    def train_ids(self):
        tids = range(0, self.train_nb)
        tids = list(set( tids) - set(self.bad_train_ids) )  # Remove bad ids
        tids.sort()
        return tids
                    
    @property 
    def test_ids(self):
        return range(0, self.test_nb)
    
    def path(self, name, **kwargs):
        """Return path to various source files"""
        path = self.source_paths[name].format(**kwargs)
        return path        

    def counts(self) :
        counts = {}
        fn = self.path('counts')
        with open(fn) as f:
            f.readline()
            for line in f:
                tid_counts = list(map(int, line.split(',')))
                counts[tid_counts[0]] = tid_counts[1:]
        return counts

    def load_train_image(self, train_id): 
        fn = self.path('train', iid=train_id)
        img = Image.open(fn)
        return img
   
    def load_dotted_image(self, train_id):
        fn = self.path('dotted', iid=train_id)
        img = Image.open(fn)
        return img
            

    def coords(self, train_id):
        # Empirical constants
        MIN_DIFFERENCE = 30
        MIN_AREA = 15
        MAX_AREA = 100
        MAX_AVG_DIFF = 50
       
        src_fn = self.path('train', iid=train_id)
        src_img = np.asarray(Image.open(src_fn), dtype = np.float)
    
        dot_fn = self.path('dotted', iid=train_id)
        dot_img = np.asarray(Image.open(dot_fn), dtype = np.float)

        img_diff = np.abs(src_img-dot_img)
        
        # Detect bad data. If train and dotted images very different somethings wrong.
        img_diff[dot_img==0] = 0              # Mask out black masks in dotted images
        avg_diff = img_diff.sum() / (img_diff.shape[0] * img_diff.shape[1])
        if avg_diff > MAX_AVG_DIFF: return None
        
        img_diff[dot_img==0] = 255              # Mask out black masks in dotted images
        img_diff = np.max(img_diff, axis=-1)
           
        img_diff[img_diff<MIN_DIFFERENCE] = 0
        img_diff[img_diff>=MIN_DIFFERENCE] = 255

        # debug
        #img = Image.fromarray(img_diff.astype(np.uint8) )
        #img.save('img_diff.png')

        sealions = []

        contours = cv2.findContours(img_diff.astype(np.uint8), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        for cnt in contours :
            area = cv2.contourArea(cnt) 
            if(area>MIN_AREA and area<MAX_AREA) :
                p = Polygon(shell=cnt[:, 0, :])
                x,y = p.centroid.coords[0]
                x = int(x)
                y = int(y)
                cls = self.class_from_color(dot_img[y,x])
                if cls is None: continue
                #print(cls, x,y)
                sealions.append ( (cls, x, y) )

        return sealions


# Count sea lion dots and compare to truth from train.csv
sld = SeaLionData()
true_counts = sld.counts()

'''
print('train_id','true_counts','counted_dots', 'difference', sep='\t')
for train_id in sld.trainshort_ids:
#for train_id in (6,):   
    coords = sld.coords(train_id)
    if coords is None: 
        print(train_id, '\tBAD DATA ') 
        continue
    counts = [0,0,0,0,0]
    for c in coords :
        counts[c[0]] +=1
    print(train_id, true_counts[train_id], counts, np.array(true_counts[train_id]) - np.array(counts) , sep='\t' )
'''

index = 5

image = cv2.cvtColor(cv2.imread(sld.path('train', iid=index)), cv2.COLOR_BGR2RGB)
image_dot = cv2.cvtColor(cv2.imread(sld.path('dotted', iid=index)), cv2.COLOR_BGR2RGB)

img = image[1350:1900, 3000:3400]
img_dot = image_dot[1350:1900, 3000:3400]

#img_c = np.copy(img)

diff = cv2.absdiff(img_dot, img)
gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
ret,th1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
plt.figure(figsize=(16,8))
plt.imshow(th1, 'gray')

cnts = cv2.findContours(th1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
print("Sea Lions Found: {}".format(len(cnts)))

for (i, c) in enumerate(cnts):
	((x, y), _) = cv2.minEnclosingCircle(c)
	cv2.putText(diff, "{}".format(i + 1), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
	cv2.drawContours(diff, [c], -1, (0, 255, 0), 2)

plt.figure(figsize=(16,8))
plt.imshow(diff)

f, ax = plt.subplots(3,1,figsize=(18,35))
(ax1, ax2, ax3) = ax.flatten()
ax1.imshow(img_dot)
ax2.imshow(th1, 'gray')
ax3.imshow(diff)
plt.show()
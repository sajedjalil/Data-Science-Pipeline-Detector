from __future__ import print_function
from __future__ import division
import sys
import os

import pandas as pd
import numpy as np
import glob

import shapely
import shapely.geometry
from shapely.geometry import Polygon
import skimage
import skimage.io
import skimage.measure

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
            (243,8,5),          # red
            (244,8,242),       # magenta
            (87,46,10),          # brown 
            (25,56,176),        # blue
            (38,174,21),        # green
            )

        self.actual_colors = {c : [] for c in self.class_colors}
        
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
        self.bad_train_ids = set([
            3, 7, 9, 21, 30, 34, 71, 81, 89, 97, 151, 184, 215, 234, 242, 
            268, 290, 311, 331, 344, 380, 384, 406, 421, 469, 475, 490, 499, 
            507, 530, 531, 605, 607, 614, 621, 638, 644, 687, 712, 721, 767, 
            779, 781, 794, 800, 811, 839, 840, 869, 882, 901, 903, 905, 909, 
            913, 927, 946])
        
    @property
    def trainshort_ids(self):
        return [x for x in range(0, 11) if x not in self.bad_train_ids] 
        
    @property 
    def train_ids(self):
        return [x for x in range(self.train_nb) if x not in self.bad_train_ids]
                    
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
        img = skimage.io.imread(fn)
        return img
   
    def load_dotted_image(self, train_id):
        fn = self.path('dotted', iid=train_id)
        img = Image.open(fn)
        return img
            

    def coords(self, train_id):
        # Empirical constants
        MIN_DIFFERENCE = 16
        MIN_AREA = 9
        MAX_AREA = 100
        MAX_AVG_DIFF = 50
        MAX_COLOR_DIFF = 32
       
        src_fn = self.path('train', iid=train_id)
        src_img = np.asarray(skimage.io.imread(src_fn), dtype = np.float)
    
        dot_fn = self.path('dotted', iid=train_id)
        dot_img = np.asarray(skimage.io.imread(dot_fn), dtype = np.float)

        img_diff = np.abs(src_img-dot_img)
        
        # Detect bad data. If train and dotted images very different somethings wrong.
        img_diff[dot_img==0] = 0              # Mask out black masks in dotted images
        avg_diff = img_diff.sum() / (img_diff.shape[0] * img_diff.shape[1])
        if avg_diff > MAX_AVG_DIFF: return None
        
        img_diff[dot_img.max(axis=-1)==0] = 255              # Mask out black masks in dotted images
        img_diff = np.max(img_diff, axis=-1)
           
        img_diff[img_diff<MIN_DIFFERENCE] = 0
        img_diff[img_diff>=MIN_DIFFERENCE] = 255

        # debug
        #img = Image.fromarray(img_diff.astype(np.uint8) )
        #img.save('img_diff.png')

        sealions = []
        
        for cls, color in enumerate(self.class_colors):
            color_array = np.array(color)[None, None, :]
            has_color = np.sqrt(np.sum(np.square(dot_img * (img_diff > 0)[:,:,None] - color_array), axis=-1)) < MAX_COLOR_DIFF 
            contours = skimage.measure.find_contours(has_color.astype(float), 0.5)
        
            for cnt in contours :
                p = Polygon(shell=cnt)
                area = p.area 
                if(area > MIN_AREA and area < MAX_AREA) :
                    y, x = p.centroid.coords[0]
                    x = int(x)
                    y = int(y)
                    c = 0
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            c += dot_img[y + dy, x + dx] 
                    self.actual_colors[color].append(c / 9)
                    sealions.append ( (cls, x, y) )

        return sealions


# Count sea lion dots and compare to truth from train.csv
sld = SeaLionData()
true_counts = sld.counts()





print('train_id','true_counts','counted_dots', 'difference', sep='\t')
for train_id in sld.trainshort_ids:
    try:
        coords = sld.coords(train_id)
    except IOError:
        print("Could not load", train_id)
        continue
    if coords is None: 
        print(train_id, '\tBAD DATA ') 
        continue
    counts = [0,0,0,0,0]
    for c in coords :
        counts[c[0]] +=1
    print(train_id, true_counts[train_id], counts, np.array(true_counts[train_id]) - np.array(counts) , sep='\t' )

for color, l in sld.actual_colors.items():
    print(color, np.mean(l, axis=0))

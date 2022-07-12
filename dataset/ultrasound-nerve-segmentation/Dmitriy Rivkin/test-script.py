# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
import pickle

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# test_img = os.listdir("../input/test")
# img1 = cv2.imread("../input/test/" + test_img[0])
# cv2.imwrite('test_image.tiff',img1)

rows = 420
cols = 580
image_size = rows*cols

with open("../input/train_masks.csv",'r') as f:
    dat = f.read()

dat = dat.split('\n')[1:-1]
total = float(len(dat))

def size_of_mask(mask_string):
    n = mask_string.split(' ')
    ms = 0
    for i in range(1,len(n),2):
        ms = ms + float(n[i])
    return ms
    
def full_plane_score(mask_size):
    if (mask_size > image_size):
        print(mask_size)
    return 2*mask_size/(mask_size+image_size)


#this is what you would get on the training set if you were able to perfectly distinguish which images had nerves
#and ones that didn't, and on the ones that did you guessed that the whole image is a nerve
avg_score = 0
for d in dat:
    p = d.split(',')
    if len(p[2])==0:
        avg_score = avg_score + 1/total
    else:
        # print(full_plane_score(size_of_mask(p[2]))/total)
        avg_score = avg_score + full_plane_score(size_of_mask(p[2]))/total
print(avg_score)

#this is what you would get on the training set if for all images that had nerves, you could put perfect bounding boxes

def get_2d_coord(point):
    row = point % rows
    col = point % cols
    # print(row,col)
    return(row,col)

#compute size of bounding box for mask
def bounding_box_size(mask_string):
    n = mask_string.split(' ')
    minx = cols
    miny = rows
    maxx = 0
    maxy = 0
        
        
    for i in range(0,len(n),2):
        y,x = get_2d_coord(int(n[i]))
        if x < minx:
            minx = x
        if y < miny:
            miny = y
        if x > maxx:
            maxx = x
        if y > maxy:
            maxy = y
        
        y,x = get_2d_coord(int(n[i])+int(n[i+1]))
        if x < minx:
            minx = x
        if y < miny:
            miny = y
        if x > maxx:
            maxx = x
        if y > maxy:
            maxy = y
    return (maxx-minx)*(maxy-miny)

def bbox_score(box_size,mask_size):
    if mask_size > box_size:
        print(box_size)
    return 2.0*mask_size/(box_size+mask_size)
    
avg_score = 0
for d in dat:
    p = d.split(',')
    if len(p[2])==0:
        avg_score = avg_score + 1/total
    else:
        # print(full_plane_score(size_of_mask(p[2]))/total)
        avg_score = avg_score + bbox_score(bounding_box_size(p[2]),size_of_mask(p[2]))/total
print(avg_score)



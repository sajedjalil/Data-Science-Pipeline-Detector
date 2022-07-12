# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


#The script wont run as a kernel, run it locally

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
%matplotlib inline
import matplotlib.pyplot as plt
import time


import matplotlib.image as mpimg
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.

# to get files locations
train_dir = '../train'
test_dir = '../test'

train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)

# example how feature detector from opencv works
def image_detect_and_compute(detector, img_name):
    """Detect and compute interest points and their descriptors."""
    img = cv2.imread(os.path.join(train_dir, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(img, None)
    return img, kp, des

def image_detect_and_compute_path(detector, img_name, path):
    """Detect and compute interest points and their descriptors."""
    img = cv2.imread(os.path.join(path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(img, None)
    return img, kp, des    

def draw_image_matches(detector, img1_name, img2_name, nmatches=10):
    """Draw ORB feature matches of the given two images."""
    img1, kp1, des1 = image_detect_and_compute(detector, img1_name)
    img2, kp2, des2 = image_detect_and_compute(detector, img2_name)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x: x.distance) # Sort matches by distance.  Best come first.

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:nmatches], img2, flags=2) # Show top 10 matches
    plt.figure(figsize=(7, 7))
    plt.title(type(detector))
    plt.imshow(img_matches); plt.show()
    

orb = cv2.ORB_create()
draw_image_matches(orb, train_files[64], train_files[64])

%%time
# img2 test

i = 0
max_iter = 10000 #takes less than 1 minute to compare 1 image with 10k others

distes = []
img2, kp2, des2 = image_detect_and_compute_path(detector, test_files[0], test_dir)
for train_file in train_files:
    if i <= max_iter:
        img1, kp1, des1 = image_detect_and_compute(detector, train_file)
        bf = cv2.S(cy, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x: x.distance) # sorting top matches
        dist = 0
        for match  in matches[0:11]: 
            dist = dist + match.distance
            distes.append(dist/12)
            i = i + 1
# to sort and create dict with dist - slow way
b = {i: distes[i] for i in range(0, len(distes), 1)} 
print (sorted(b.items(), key=lambda x: x[1])[0:5] # to get TOP 5 matches

img1 = cv2.imread(train_dir + '/' + train_files[3769])
plt.imshow(img1)

img2 = cv2.imread(test_dir + '/' + test_files[0])
plt.imshow(img2)
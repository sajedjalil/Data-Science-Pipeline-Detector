# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# This small script goes through all images and creates the csv file with the 
# feature indicating whether floorplan was provided for a given listing id
# it is based on an idea that (may be) availability of the floor plan increases 
# attractiveness of the listing
# colors are binned into the buckets and if any bucket contains more than 40% of all pixels
# then it is considered to be a floorplan
# this method seems to have a rather high precision, but i am not sure about recall
# also, resulting csv will have a couple of duplicate listing_id-s from some reason
# so, dedup before using

import os
import cv2
import IPython.display
from skimage import data
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
import csv
import pandas as pd

output_list = []
total_listings_count = 124011
images_root_dir = r'e:\\images\\'

listings_processed = 0
for root, dir, files in os.walk(images_root_dir):
    listing_id = root[-7:]
    is_floorplan = False
    #print("processing listing id",listing_id )
    for filename in files:
        file_with_path = root +r'\\'+ filename
        img = cv2.imread(file_with_path,0)
        hist = cv2.calcHist([img],[0],None,[256],[0,256]) # get the image histogram
        hist=[hist[i][0] for i in range(0,len(hist))]  # convert histogram into simple list
        
        # normalize the histogram - from pixel count to share
        tot_pix =sum(hist) 
        hist = hist / tot_pix
        
        # get the number of bins with share of pixels greater than 0.4 - indicative of the floor plan
        numcol = len(hist[hist>0.5])
        
        #print('processing file',filename,hist[0])
        is_floorplan = is_floorplan|(numcol >0)
    if is_floorplan:
        result = [listing_id, 1]
    else:
        result = [listing_id, 0]
        
    output_list.append(result)
    listings_processed +=1
    if listings_processed%500 == 0:
        print("processed percentage: ", (listings_processed/total_listings_count)*100)
        print("length of output_list: ", len(output_list))
        
df = pd.DataFrame(output_list[1:])
df.to_csv("floorplan_provided1.csv", index=False, header=False)
        

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 as opencv
import matplotlib.pyplot as plot
from glob import glob
from skimage import io

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


#print(check_output(["", "../input/test-jpg-v2"]).decode("utf8"))
paths_jpg = []
paths_tif =  []
train_images_jpg = {}
train_images_tif = {}
image_paths_jpg = {}
image_paths_tif = {}

train_file = pd.read_csv('../input/train_v2.csv')[0:186];

def readTags( ):
    print(len(train_file))
    for index, row in train_file.iterrows():
        print (row["image_name"], row["tags"])

def readTrainInput():
    global train_file
    #print (train_file)
    

def readTrainImagesJpg():
    image_paths = sorted(glob('../input/train-jpg/*.jpg'))[0:186]
    for path in image_paths:
        paths_jpg.append(path)
        image_paths_jpg[path] = path.replace("../input/train-jpg/","")
        img = opencv.imread(path)
        train_images_jpg[path.replace("../input/train-jpg/","")] = img
    #print (train_images_jpg)
    
    
def readTrainImagesTif():
    image_paths = sorted(glob('../input/train-tif-v2/*.tif'))[0:186]
    for path in image_paths:
        paths_tif.append(path)
        image_paths_tif[path] = path.replace("../input/train-tif-v2/","")
        img = opencv.imread(path)
        train_images_tif[path.replace("../input/train-tif-v2/","")] = img
    #print (image_paths_tif)    
    
    
def saveImageFileJpg(filename,img):    
    opencv.imwrite(filename+'.jpg',img)
    
def saveImageFileTif(filename,img):    
    opencv.imwrite(filename+'.tif',img)  
    
    
 
    
    
    
readTrainInput()    
readTrainImagesJpg()
readTrainImagesTif()
readTags()

#print (image_paths_tif)
#print (image_paths_jpg)
saveImageFileJpg("image_185",train_images_jpg[image_paths_jpg[paths_jpg[184]]])
saveImageFileTif("image_185",train_images_tif[image_paths_tif[paths_tif[184]]])





# Any results you write to the current directory are saved as output.
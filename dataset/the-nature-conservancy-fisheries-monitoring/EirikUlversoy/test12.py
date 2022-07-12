# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from scipy import ndimage
import cv2
from subprocess import check_output

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (32, 32), cv2.INTER_LINEAR)
    return resized

def read_fish_folder(fish_foldername):
    filenames = check_output(["ls", train_path+fish_foldername]).decode("utf8").strip().split('\n')
    #filenames = check_output(["ls", train_path+fish_foldername])
    new_filenames = []
    for filename in filenames:
        filename = train_path+fish_foldername+"/"+filename
        new_filenames.append(filename)
    for filename in new_filenames:
        print(filename)
     
    return new_filenames
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train_path = "../input/train/"
sub_folders = check_output(["ls", train_path]).decode("utf8").strip().split('\n')
print(sub_folders)


print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
fish_filenames = []
fish_images = []
fish_ids = []

for folder in sub_folders:
    filenames = read_fish_folder(folder)
    fish_filenames.append(filenames)
    
for fish_category in fish_filenames:
    for fish in fish_category:
        fish_images.append(get_im_cv2(fish))
        

#print(fish_filenames[0][0])
#sample_fish = cv2.imread(fish_filenames[0][0])
#print(sample_fish)
#cv2.imshow('image',sample_fish)

    

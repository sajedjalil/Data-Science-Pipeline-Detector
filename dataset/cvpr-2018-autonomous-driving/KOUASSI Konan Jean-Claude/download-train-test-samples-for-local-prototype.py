# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

__author__ = 'kjeanclaude: https://kaggle.com/kjeanclaude'
#### Inspired by https://www.kaggle.com/lantingguo/get-a-few-images-and-labels-for-local-prototype
#### Improved with test set sample too, data reorganization and zip files creation for the output.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Get a few images and labels for local prototype. Because it will cost me 2 or 3 days to download train_color.zip.

from pathlib import Path
from skimage.io import imread, imsave

train_color_dir = Path("../input/train_color")
train_label_dir = Path("../input/train_label")
test_dir = Path("../input/test")

train = 'train/train_color/'
label_dir = 'train/train_label/'
test = 'test/'

os.makedirs(train)
os.makedirs(label_dir)
os.makedirs(test)

print('Starting with training set ... ')
num_sample = 10
for img_file in os.listdir(train_color_dir)[:num_sample]:
    picture_file = os.path.join(train, img_file)
    img = imread(train_color_dir.joinpath(img_file))
    plt.imsave(picture_file, img)
    label_file = img_file[:-4]+"_instanceIds.png"
    picture_file2 = os.path.join(label_dir, label_file)
    label = imread(train_label_dir.joinpath(label_file))
    plt.imsave(picture_file2, label)
    
print('Continue with test set ... ')
for img_file in os.listdir(test_dir)[:num_sample]:
    picture_file = os.path.join(test, img_file)
    img = imread(test_dir.joinpath(img_file))
    plt.imsave(picture_file, img)


#!zip -r 'train.zip' 'train'
#!zip -r 'test.zip' 'test'
import shutil
#shutil.make_archive('archive', 'zip', target_path, file_to_zip)
print('Compressing the train fold ... ')
shutil.make_archive("train", "zip", "train")
print('Compressing the test fold ... ')
shutil.make_archive("test", "zip", "test")

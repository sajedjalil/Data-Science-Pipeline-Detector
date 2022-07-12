# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# coding: utf-8
# import data
import os
import matplotlib.image as image
import matplotlib.pyplot as plt

PLANET_KAGGLE_ROOT = "../input"
PLANET_KAGGLE_TIFF_DIR = PLANET_KAGGLE_ROOT + "/train-tif/"
PLANET_KAGGLE_LABEL_CSV = PLANET_KAGGLE_ROOT + "/train.csv"

# load label data
# read our data and take a look at what we are dealing with
train_csv = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)
print(train_csv.head())

# load tiff data
# load files
train_tiff_sample = os.listdir(PLANET_KAGGLE_TIFF_DIR)

# pick just cloudy images for now

AVAILABLE_LABELS = [
    'agriculture', 
    'artisinal_mine', 
    'bare_ground', 
    'blooming', 
    'blow_down', 
    'clear', 
    'cloudy', 
    'conventional_mine', 
    'cultivation', 
    'habitation', 
    'haze', 
    'partly_cloudy', 
    'primary', 
    'road', 
    'selective_logging', 
    'slash_burn', 
    'water']

tags = pd.DataFrame()

for label in AVAILABLE_LABELS:
    tags[label] = train_csv.tags.apply(lambda x: np.where(label in x, 1, 0))

train = pd.concat([train_csv, tags], axis=1)
filtered_samples = [sample[0:-4] for sample in train_tiff_sample]

# get labels for these files
sample_labels = train[train['image_name'].isin(filtered_samples)]
sample_labels.reset_index(inplace=True)
X = sample_labels[['cloudy']]

from glob import glob
from skimage import io
image_paths = sorted(glob('../input/train-tif/*.tif'))[0:1000]
imgs = [io.imread(path) for path in image_paths]


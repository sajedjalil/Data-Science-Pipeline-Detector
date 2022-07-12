# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from PIL import Image

metadata = pd.read_csv('../input/train.csv')

# three helpful attributes for the data preprocessing (will be a Series objects in the DataFrame soon)
height_list = list()
width_list = list()
is_grayscale_list = list()

# fill the lists
for ix, path in metadata['Image'].iteritems():
    with Image.open(os.path.join('../input/train/', path)) as im:
        gsc = None
        h, w = im.size
        m = im.mode
        height_list.insert(ix, h)
        width_list.insert(ix, w)
        gsc = 0 if m == 'RGB' else 1
        is_grayscale_list.insert(ix, gsc)

# check if they are consistent
assert(len(height_list) == len(width_list) == len(is_grayscale_list))
        
# add them to our DataFrame
metadata['height'] = height_list
metadata['width'] = width_list
metadata['is_grayscale'] = is_grayscale_list

# quick look at the new attributes
print(metadata.describe())

# MAIN GOAL HERE: save file for the future use
metadata.to_csv('metadata.csv', header=True, index=False)

print('File saved successfully, now you may use metadata.csv instead of ../input/train.csv')
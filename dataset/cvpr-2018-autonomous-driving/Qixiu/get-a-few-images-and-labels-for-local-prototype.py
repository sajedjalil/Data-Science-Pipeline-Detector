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

# Get a few images and labels for local prototype. Because it will cost me 2 or 3 days to download train_color.zip.

from pathlib import Path
from skimage.io import imread, imsave

train_color_dir = Path("../input/train_color")
train_label_dir = Path("../input/train_label")

num_sample = 4
for img_file in os.listdir(train_color_dir)[:num_sample]:
    img = imread(train_color_dir.joinpath(img_file))
    imsave(img_file, img)
    label_file = img_file[:-4]+"_instanceIds.png"
    label = imread(train_label_dir.joinpath(label_file))
    imsave(label_file, label)

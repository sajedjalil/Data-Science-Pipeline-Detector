# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
from PIL.ExifTags import TAGS
import os

base_path='../input'

for d in os.listdir(f'{base_path}/images_sample'):
    if os.path.isdir(f'{base_path}/images_sample/{d}'):
        for f in os.listdir(f'{base_path}/images_sample/{d}'):
            img = Image.open(f'{base_path}/images_sample/{d}/{f}')
            print(f'Listing_image_id: {d}')
            print('Image size: %s' % (img.size[0] * img.size[1]))
            tags = img._getexif()
            if tags:
                for (k, v) in img._getexif().items():
                    if TAGS.get(k) not in ('MakerNote', None):
                        print('%s = %s' % (TAGS.get(k), v))
            print('\n')
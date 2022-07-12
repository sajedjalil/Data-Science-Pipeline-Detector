# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import gc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-

def CreateImageFile(row):
    file_name = '/kaggle/work/image/' + row['image_id'] + '.png'
    image_array = row.iloc[1:].values.reshape(137, 236).astype(np.uint8)
    pil_img = Image.fromarray(image_array)
    pil_img.save(file_name)
    del image_array
    del pil_img
    gc.collect()
    

def LoadImage(image_path_list):
    for item in image_path_list:
        print('path : {}'.format(item))
        image_df = pd.read_parquet(item)
        row_num = len(image_df)
        for index in range(row_num):
            CreateImageFile(image_df.loc[index])
        del image_df
        gc.collect()
        
def LoadTrainingImage():
    input_path_list = [
            '/kaggle/input/bengaliai-cv19/train_image_data_0.parquet',
            '/kaggle/input/bengaliai-cv19/train_image_data_1.parquet',
            '/kaggle/input/bengaliai-cv19/train_image_data_2.parquet',
            '/kaggle/input/bengaliai-cv19/train_image_data_3.parquet',
        ]

    LoadImage(input_path_list)
    
def main():
    os.makedirs('/kaggle/work/image/')
    LoadTrainingImage()
    
if __name__ == '__main__':
    main()

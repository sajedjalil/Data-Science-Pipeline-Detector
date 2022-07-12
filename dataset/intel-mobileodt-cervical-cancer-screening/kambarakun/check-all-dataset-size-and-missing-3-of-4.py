#!/usr/bin/env python
# -*- coding: utf-8 -*-


# This script checks the dataset's image shape, and jpeg file's error or warning.
# This script checks all dataset (stage 1, train, test, additional)
# This script is part 3 of 4, for computing in 1200 sec.

# Shape includes size (height, width) and number of color channel (RGB = 3)
# shape_1 = height, shape_2 = width, shape_3 = 3

# error   =  blank 0 byte file
# additional/Type_2/2845.jpg
# additional/Type_2/5892.jpg
# additional/Type_2/5893.jpg

# warning = Premature end of JPEG file
# train/Type_1/1339.jpg has about 55% data of the image size. This file can't be used.
# additional/Type_1/3068.jpg has about 78% data of the image size. This file can be used.
# additional/Type_2/7.jpg has about 75% data of the image size. This file maybe be used.

import os
import platform
import cv2
import numpy
import PIL.Image
import pandas


# Get the list of all jpeg files in directory
def get_list_abspath_img(abspath_dataset_dir):
    list_abspath_img = []
    for str_name_file_or_dir in os.listdir(abspath_dataset_dir):
        if ('.jpg' in str_name_file_or_dir) == True:
            list_abspath_img.append(os.path.join(abspath_dataset_dir, str_name_file_or_dir))
    list_abspath_img.sort()
    return list_abspath_img


if 'c001' in platform.node():
    # platform.node() => 'c001' or like 'c001-n030' on Colfax
    abspath_dataset_dir_train_1 = '/data/kaggle/train/Type_1'
    abspath_dataset_dir_train_2 = '/data/kaggle/train/Type_2'
    abspath_dataset_dir_train_3 = '/data/kaggle/train/Type_3'
    abspath_dataset_dir_test    = '/data/kaggle/test/'
    abspath_dataset_dir_add_1   = '/data/kaggle/additional/Type_1'
    abspath_dataset_dir_add_2   = '/data/kaggle/additional/Type_2'
    abspath_dataset_dir_add_3   = '/data/kaggle/additional/Type_3'
elif '.local' in platform.node():
    # platform.node() => '*.local' on my local MacBook Air
    abspath_dataset_dir_train_1 = '/Volumes/TRANSCEND_G/intel-mobileodt-cervical-cancer-screening/train/Type_1'
    abspath_dataset_dir_train_2 = '/Volumes/TRANSCEND_G/intel-mobileodt-cervical-cancer-screening/train/Type_2'
    abspath_dataset_dir_train_3 = '/Volumes/TRANSCEND_G/intel-mobileodt-cervical-cancer-screening/train/Type_3'
    abspath_dataset_dir_test    = '/Volumes/TRANSCEND_G/intel-mobileodt-cervical-cancer-screening/test/'
    abspath_dataset_dir_add_1   = '/Volumes/TRANSCEND_G/intel-mobileodt-cervical-cancer-screening/additional/Type_1'
    abspath_dataset_dir_add_2   = '/Volumes/TRANSCEND_G/intel-mobileodt-cervical-cancer-screening/additional/Type_2'
    abspath_dataset_dir_add_3   = '/Volumes/TRANSCEND_G/intel-mobileodt-cervical-cancer-screening/additional/Type_3'
else:
    # For kaggle's kernels environment (docker container?)
    abspath_dataset_dir_train_1 = '/kaggle/input/train/Type_1'
    abspath_dataset_dir_train_2 = '/kaggle/input/train/Type_2'
    abspath_dataset_dir_train_3 = '/kaggle/input/train/Type_3'
    abspath_dataset_dir_test    = '/kaggle/input/test/'
    abspath_dataset_dir_add_1   = '/kaggle/input/additional/Type_1'
    abspath_dataset_dir_add_2   = '/kaggle/input/additional/Type_2'
    abspath_dataset_dir_add_3   = '/kaggle/input/additional/Type_3'

list_abspath_img_train_1 = get_list_abspath_img(abspath_dataset_dir_train_1)
list_abspath_img_train_2 = get_list_abspath_img(abspath_dataset_dir_train_2)
list_abspath_img_train_3 = get_list_abspath_img(abspath_dataset_dir_train_3)
list_abspath_img_test    = get_list_abspath_img(abspath_dataset_dir_test)
list_abspath_img_add_1   = get_list_abspath_img(abspath_dataset_dir_add_1)
list_abspath_img_add_2   = get_list_abspath_img(abspath_dataset_dir_add_2)
list_abspath_img_add_3   = get_list_abspath_img(abspath_dataset_dir_add_3)

list_abspath_img_train   = list_abspath_img_train_1 + list_abspath_img_train_2 + list_abspath_img_train_3
list_abspath_img_add     = list_abspath_img_add_1   + list_abspath_img_add_2   + list_abspath_img_add_3


# Header of output
pandas_header = ['abspath', 'shape_1', 'shape_2', 'shape_3', 'error', 'warning']
pandas_data   = []

# Join lists of abspath
list_abthpath = list_abspath_img_train + list_abspath_img_test + list_abspath_img_add

# Spilit list by kernel for computing in 1200 sec.
list_abthpath = list_abthpath[4500:6750]

# Check all jpeg file
for abspath_img in (list_abthpath):
    # Open file via OpenCV
    img = cv2.imread(abspath_img)

    # 0 byte file can be imread(), but does't have shape attribure
    if hasattr(img, 'shape') == False:
        # Add shape info to pandas_data (header = abspath,shape_1,shape_2,shape_3,error,warning)
        pandas_data.append([abspath_img, 'Nan', 'Nan', 'Nan', 'error', ''])
        continue
    else:
        # img.shape returns like (640, 480, 3) -> 640,480,3 -> [640, 480, 3]
        list_shape = str(img.shape).replace('(', '').replace(' ', '').replace(')', '').split(',')
        warning    = ''
        try:
            # Check Premature end of JPEG file
            # If premature jpeg file, TypeError: long() argument must be a string or a number, not 'JpegImageFile'
            cv2.cvtColor(numpy.array(PIL.Image.open(open(abspath_img, 'rb')), dtype=numpy.uint8), cv2.COLOR_RGB2BGR)
        except:
            # Maybe(!?) Premature end of JPEG file
            warning = 'warning'
        finally:
                    # Add shape info to pandas_data (header = abspath,shape_1,shape_2,shape_3,error,warning)
                    pandas_data.append([abspath_img, list_shape[0], list_shape[1], list_shape[2], '', warning])

pandas.DataFrame(pandas_data, columns = pandas_header).to_csv('check_all_dataset_3.csv', index=False)

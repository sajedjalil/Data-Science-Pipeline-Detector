# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn 
import scipy
import glob #path patter expression
import os
import cv2
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.



imgs_list = pd.read_csv('../input/driver_imgs_list.csv')
class_number = imgs_list['classname'].str.extract('(\d)', expand =False) #convert class name to number


train_files = [f for f in glob.glob("../input/train/*/*.jpg")]
labels = pd.DataFrame({'label': ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'], 'class_name':['safe driving','texting - right', 'talking on the phone - right', 'texting - left', 'talking on the phone - left', 'operating the radio', 'drinking', 'reaching behind', 'hair and makeup', 'talking to passenger' ]})


data = pd.DataFrame({'file':train_files})
data['label'] = data['file'].str.extract('(c.)', expand=True)

train = data.merge(labels, on = 'label', how = 'left')
print(train.iloc[0,0])



im = cv2.imread(train.iloc[0,0])
img = cv2.resize(im, (640, 480))
gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap = plt.get_cmap('gray'))

#cv2.imshow('img', img)
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


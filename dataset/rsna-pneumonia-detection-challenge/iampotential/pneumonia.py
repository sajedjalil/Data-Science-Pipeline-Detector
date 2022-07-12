# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from skimage.transform import resize
import pandas as pd
from matplotlib import pyplot as plt
import pydicom, numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize

df = pd.read_csv('../input/stage_1_detailed_class_info.csv')
patientId = df['patientId'][1000]
dcm_file = '../input/stage_1_train_images/%s.dcm' % patientId
dcm_data = pydicom.read_file(dcm_file)



df['class'].value_counts(normalize=True)
labels = df.iloc[:10000]
labels.replace(to_replace={'No Lung Opacity / Not Normal','Lung Opacity','Normal'},value={2,1,0},inplace=True)
labels.replace(to_replace=2,value=1,inplace=True)
CLass = labels['class']
CLass.dtype
labels = np.array((CLass),dtype='float64')
images = []

for i in df.patientId.unique():
    images.append('../input/stage_1_train_images/%s.dcm' % i)
im_in_pixels = []

for i in images[:10000]:
    im_in_pixels.append(pydicom.read_file(i))

#plt.imshow(im_in_pixels[31].pixel_array,'Greys')
im_data = []
for i in range(1,10000):
    im_data.append(im_in_pixels[i].pixel_array)

im_data = []
for i in range(0,10000):
    im_data.append(im_in_pixels[i].pixel_array)
plt.imshow(im_data[14],'Greys')
resizz = []
for i in range(0,10000):
    resizz.append(resize(im_data[i],(28,28)))
len(resizz[14])
plt.imshow(resizz[11],'Greys')
labels.dtype
target= df.iloc[:10000]
targets = CLass.iloc[:10000]
three_D = []
for i in range(0,10000):
    three_D.append(resizz[i].reshape(28, 28,1))
stacked = np.dstack(three_D)

stacked.reshape(28,28,10000)
from sklearn import preprocessing
from sklearn import model_selection
target = targets


stacked.reshape(10000,784)
stacked_2 = stacked.reshape(10000,784)

x_test,x_train,y_test,y_train = model_selection.train_test_split(stacked_2,targets,test_size=.7,random_state=11)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
fit = clf.fit(x_train,y_train)
score = clf.score(x_test,y_test)
print(score)

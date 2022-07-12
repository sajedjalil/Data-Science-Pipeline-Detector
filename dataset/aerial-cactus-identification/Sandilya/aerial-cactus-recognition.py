# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
from keras import Sequential
from keras.layers import Dense,Flatten
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
#print(os.listdir("../input/train"))
#print(os.listdir("../input/test/test"))
path="../input"

train_path=path+"/train/train/"
test_path=path+"/test/test/"

train_images=os.listdir(train_path)
test_images=os.listdir(test_path)

train_set=pd.read_csv(path+'/train.csv')
train_labels=train_set['has_cactus']

train_images_all=[np.array(Image.open(train_path+image).convert('1'),dtype='int32') for image in train_images]
test_images_all=[np.array(Image.open(test_path+image).convert('1'),dtype='int32') for image in test_images]

train_images_all=np.array(train_images_all)
test_images_all=np.array(test_images_all)

print(train_images_all.shape)
print(test_images_all.shape)
# train_images_all=train_images_all
# test_images_all=test_images_all

model=Sequential()
model.add(Dense(32,input_shape=(32,32)))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=train_images_all,y=train_labels,epochs=10)

test_data=pd.read_csv(path+'/sample_submission.csv')

test_labels= [np.argmax(each) for each in model.predict(test_images_all)]

#print(test_labels)

test_data['has_cactus']=test_labels

test_data.to_csv('sample_submission.csv',index=False)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
from PIL import ImageFile
import random
import os
def accessImage(path,resize):
    image = Image.open(path)
    image = image.resize((512, 512))
    image = np.array(image)
    return image
base_path = '/kaggle/input/siim-isic-melanoma-classification'
train_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
test_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'
img_stats_path = '/kaggle/input/melanoma2020imgtabular'

model = models.Sequential()
model.add(layers.Conv2D(16, (8, 8), strides=(8,8) ,activation='relu', input_shape=(512, 512, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (8, 8),strides=(2,2) ,activation='relu'))
model.add(layers.Conv2D(32, (8, 8), activation='relu'))
model.add(layers.Flatten(input_shape=(6, 6,32)))
model.add(layers.Dense(2,activation = 'softmax')) 
model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
model.summary()
trainF = pd.read_csv(os.path.join(base_path, 'train.csv'))
breakFlag = 0
for y in range(350):
    lastIndex = (y+1)*100
    if lastIndex > len(trainF['image_name']):
        lastIndex = len(trainF['image_name'])-1
        breakFlag = 1
    train = trainF[y*100:lastIndex]
    trainIm = train['image_name']
    train_images = trainIm.values.tolist()
    train_images = [os.path.join( i + ".jpg") for i in train_images]
    trImages = []
    for (imname,target) in zip(train_images,train['target']):
            img = accessImage(train_img_path+imname,[512,512])
            if target == 1:
                for i in range(19):
                    trImages.append([img,target])
            trImages.append([img,target])
    trImages,trTargets = np.array(trImages).T.tolist()
    trImages = np.asarray(trImages)
    trImages = trImages/255.
    trTargets = np.asarray(trTargets)
    if len(trImages) == 0 or len(trTargets) == 0:
        break
    history = model.train_on_batch(trImages, trTargets)
    if breakFlag == 1:
        break
del train
del trainF
del trainIm
del train_images
del trImages
del trTargets
breakFlag = 0
testF = pd.read_csv(os.path.join(base_path, 'test.csv'))
output = []
for y in range(150):
    lastIndex = (y+1)*100
    if lastIndex > len(testF['image_name']):
        lastIndex = len(testF['image_name'])
        breakFlag = 1
    test = testF[y*100:lastIndex]
    testIm = test['image_name']
    test_images = testIm.values.tolist()
    test_img = [os.path.join( i + ".jpg") for i in test_images]
    teImages = []
    for imname in test_img:
        teImages.append(accessImage(test_img_path+imname,[512,512]))
    teImages = np.asarray(teImages)
    teImages = teImages / 255.
    pred_y = model.predict_on_batch(teImages)
    for (imname,elements) in zip(test_images,pred_y):
        output.append([imname,elements[1]])
    if breakFlag == 1:
        break
del testF
del test
del testIm
del test_images
del teImages
f = open("submission.csv", "w+")
f.write('image_name,target'+os.linesep)
#for elements in output:
#    print(elements)
for element in output:
    f.write(element[0]+','+str(element[1]) + os.linesep)
f.close()
    
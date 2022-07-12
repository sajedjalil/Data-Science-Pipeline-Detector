import tensorflow as tf
from tensorflow import keras
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
from PIL import ImageFile
from PIL import ImageOps
import random
import os
import cv2
def accessImage(path,resize):
    image = Image.open(path)
    image = image.resize((512, 512))
    image = np.array(image)
    return image
def accessImageRotated(path,resize,angle):
    image = Image.open(path)
    image = image.resize((512,512))
    image = image.rotate(angle, expand=True)
    return image
base_path = '/kaggle/input/siim-isic-melanoma-classification'
train_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
test_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'
img_stats_path = '/kaggle/input/melanoma2020imgtabular'
sgd = keras.optimizers.SGD(lr=0.01,momentum=0.9,nesterov=True, clipvalue=0.5)
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (4, 4), strides=(4,4) ,activation='relu', input_shape=(512, 512, 3)))
model.add(keras.layers.Conv2D(64, (4, 4),strides=(2,2) ,activation='relu'))
model.add(keras.layers.Conv2D(64, (4, 4), activation='relu'))
model.add(keras.layers.Conv2D(64, (4,4), strides = (4,4),activation='relu'))
model.add(keras.layers.Conv2D(32, (4,4), activation = 'relu'))
model.add(keras.layers.Conv2D(16,(4,4),strides = (2,2), activation = 'relu'))
model.add(keras.layers.Flatten(input_shape=(5, 5,16)))
model.add(keras.layers.Dense(units = 400,activation = 'relu'))
model.add(keras.layers.Dense(units = 1,activation = 'sigmoid'))
model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
model.summary()
trainF = pd.read_csv(os.path.join(base_path, 'train.csv'))
breakFlag = 0
for y in range(2000):
    trTargets = []
    lastIndex = (y+1)*16
    if lastIndex > len(trainF['image_name']):
        lastIndex = len(trainF['image_name'])-1
        breakFlag = 1
    train = trainF[y*16:lastIndex]
    trainIm = train['image_name']
    train_images = trainIm.values.tolist()
    train_images = [os.path.join( i + ".jpg") for i in train_images]
    trImages = []
    for imname,t in zip(train_images,train['target']):
            image = Image.open(train_img_path+imname)
            image = image.resize((512, 512))
            img = np.array(image)
            trImages.append(img)
            trTargets.append(t)
            alter = cv2.blur(img,(512, 512))
            trImages.append(alter)
            trTargets.append(t)
            del alter
            alter = cv2.medianBlur(img,5)
            trImages.append(alter)
            trTargets.append(t)
            del alter
            alter = ImageOps.invert(image)
            trImages.append(np.array(alter))
            trTargets.append(t)
            del alter
            alter = image.rotate(90, expand=True)
            trImages.append(np.array(alter))
            trTargets.append(t)
            del alter
            alter = image.rotate(180, expand = True)
            trImages.append(np.array(alter))
            trTargets.append(t)
            del alter
            alter = image.rotate(270,expand = True)
            trImages.append(np.array(alter))
            trTargets.append(t)
            del alter
    trImages = np.asarray(trImages)
    trImages = trImages/255.
    trTargets = np.asarray(trTargets)
    class_weight = {0: 2.,1: 25.,}
    if len(trImages)==0 or len(trTargets)==0:
        break
    history = model.train_on_batch(trImages, trTargets,class_weight = class_weight)
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
for y in range(1000):
    lastIndex = (y+1)*16
    if lastIndex > len(testF['image_name']):
        lastIndex = len(testF['image_name'])
        breakFlag = 1
    test = testF[y*16:lastIndex]
    testIm = test['image_name']
    test_images = testIm.values.tolist()
    teImages = []
    for imname in test_images:
        image = Image.open(test_img_path+imname+".jpg")
        image = image.resize((512, 512))
        teImages.append(image)
    
    predictions = []
    for image in teImages:
        predictionSet = []
        img = np.array(image)
        predictionSet.append(img)
        alter = cv2.blur(img,(512, 512))
        predictionSet.append(alter)
        del alter
        alter = cv2.cv2.medianBlur(img,5)
        predictionSet.append(alter)
        del alter
        alter = ImageOps.invert(image)
        predictionSet.append(np.array(alter))
        del alter
        alter = image.rotate(90, expand=True)
        predictionSet.append(np.array(alter))
        del alter
        alter = image.rotate(180, expand = True)
        predictionSet.append(np.array(alter))
        del alter
        alter = image.rotate(270,expand = True)
        predictionSet.append(np.array(alter))
        predictionSet = np.asarray(predictionSet)
        predictionSet = predictionSet/255.
        pred_y = model.predict_on_batch(predictionSet)
        del predictionSet
        meanPrediction = 0
        for prediction in pred_y:
            meanPrediction += prediction[0]
        meanPrediction /= 7
        predictions.append(meanPrediction)
    for (imname,elements) in zip(test_images,predictions):
        output.append([imname,elements])
    if breakFlag == 1:
        break
del testF
del test
del testIm
del test_images
del teImages
del predictions
#for element in output:
#    print(element)
f = open("submission.csv", "w+")
f.write('image_name,target'+os.linesep)
for element in output:
    f.write(element[0]+','+str(element[1]) + os.linesep)
f.close()
    
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D,Dropout,Dense,Flatten
#from tensorflow.keras.utils import np_utils,to_categorical
from tensorflow.keras.optimizers import Adagrad
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import load_img,array_to_img
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
import cv2
import time

encoder = OneHotEncoder()


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

class MelanomaNet:
    @staticmethod
    def build(shape):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=shape[1:]))
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(1, activation='sigmoid'))

        return model
    
NUM_EPOCHS=50; INIT_LR=1e-2; BS=100
    
train_csv = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')


def preprocess():
    
    images=[]
    imageLabels=[]

    #print(len(train_csv))
    
    for index,row in train_csv.iterrows():
        if index>=1:
            print(index) ##For testing purposes
        images.append(cv2.resize(cv2.imread('../input/siim-isic-melanoma-classification/jpeg/train/'+row['image_name']+'.jpg'),(256,256),interpolation=cv2.INTER_CUBIC))
        #images[index] = cv2.imread('../input/siim-isic-melanoma-classification/jpeg/train/'+row['image_name']+'.jpg')
        imageLabels.append(row['target'])
    
   
    images = np.array(images)
    imageLabels = np.array(imageLabels)
   

    x_train,x_test,y_train,y_test = train_test_split(images,imageLabels,test_size=.2,random_state=42)

    #print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    encoder.fit_transform(y_train.reshape(-1,1)).toarray()
    #print(y_train)
    encoder.fit_transform(y_test.reshape(-1,1)).toarray()
    
    return x_train,x_test,y_train,y_test


def generateModel(x_train,y_train,x_test,y_test):
    model = MelanomaNet.build(x_train.shape)
    #opt = Adagrad(lr=INIT_LR,decay=INIT_LR/NUM_EPOCHS)
    model.compile(loss='binary_crossentropy',optimizer='Adadelta',metrics=['accuracy'])
    model.fit(x_train,y_train,batch_size=BS,epochs=NUM_EPOCHS, validation_data=(x_test,y_test))
    #print(model.summary())
    #model.save('melmodel.h5')


    print('Model Generated')
    
    return model

def testModel(model):
    test_csv = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

    testImages = []
    image_name=[]

    for index,row in test_csv.iterrows():
        print(index)
        testImages.append(cv2.resize(cv2.imread('../input/siim-isic-melanoma-classification/jpeg/test/'+row['image_name']+'.jpg'),(256,256),interpolation=cv2.INTER_CUBIC))
        image_name.append(row['image_name'])

    testImages = np.array(testImages)
    
    pred = model.predict(testImages)
    pred = pred.ravel()
    #print(accuracy_score(y_test,pred))
    
    print(len(pred))
    print(len(image_name))
    
    submission = pd.DataFrame({'image_name':image_name,'target':pred})
    submission.round(1)
    submission.to_csv('submission.csv',index=False)
    print(submission)

    
def main():
    
    #x_train, x_test,y_train,y_test = preprocess()
     
    #model = generateModel(x_train,y_train,x_test,y_test)

    model = load_model('melmodel.h5')

    testModel(model)


main()



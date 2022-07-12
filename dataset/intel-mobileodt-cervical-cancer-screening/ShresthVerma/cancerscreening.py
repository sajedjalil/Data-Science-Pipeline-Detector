

import numpy as np 
import pandas as pd
import os
import glob
import cv2

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

from keras.utils import np_utils



x_train=[]
y_train=[]
x_train_names=[]
x_test=[]
x_test_names=[]
y_train_labels=[]

def get_img(pathname):
    img=cv2.imread(pathname)
    img=cv2.resize(img,(32,32),cv2.INTER_LINEAR)
    return img

def load_train_data():
    folders=["Type_1","Type_2","Type_3"]
    for fld in folders:
        index=folders.index(fld)
        path=os.path.join('..','input','test','*.jpg')
        files=glob.glob(path)
        print (fld)
        print(len(files))
        for f in files:
            img=get_img(f)
            name=os.path.basename(f)
            x_train.append(img)
            y_train.append(index)
            x_train_names.append(name)
        
        
        
def load_test_data():
    path=os.path.join("..","input","test","*.jpg")
    files=glob.glob(path)
    for f in files[:100]:
        img=get_img(f)
        name=os.path.basename(f)
        x_test.append(img)
        x_test_names.append(name)

"""def create_model():
    model=Sequential()
    
    model.add(Conv2D(4,(3,3) ,activation='relu', input_shape=(3, 32, 32)))
    print(model.output_shape)
    print("hiii")
    model.add(Conv2D(4,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model
"""
def create_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 32, 32), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
    metrics=['accuracy'])
    return model

def preprocessing():
    global x_train, y_train, y_train_labels, x_test
    x_train = np.array(x_train, dtype=np.uint8)
    y_train = np.array(y_train, dtype=np.uint8)
    x_train = x_train.transpose((0, 3, 1, 2))
    x_train=x_train.astype('float32')
    x_train=x_train/255
    y_train_labels=np_utils.to_categorical(y_train)
    
    x_test=np.array(x_test,dtype=np.uint8)
    x_test=x_test.transpose((0,3,1,2))
    x_test=x_test.astype('float32')
    x_test=x_test/255
    
    
    
load_train_data()
load_test_data()
preprocessing()


    

model=create_model()

model.fit(x_train,y_train_labels,batch_size=16,epochs=30,verbose=1)
predictions=model.predict(x_test,batch_size=16,verbose=1)

print(predictions)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

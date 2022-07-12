# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
    
import os
import glob
import cv2
import time
import datetime
#-----------keras-----------------
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import __version__ as keras_version
#----------------------------------------------------------------
def load_train(inputSize):
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()
    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('..', 'input', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img=cv2.resize(cv2.imread(fl),(inputSize,inputSize),cv2.INTER_LINEAR)  
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)
    print('Convert to numpy...')
    X_train = np.array(X_train, dtype=np.uint8)
    y_train = np.array(y_train, dtype=np.uint8)
    print('Reshape...')
    X_train = X_train.transpose((0, 3, 1, 2))   
    print('Convert to float...')
    X_train = X_train.astype('float32')
    X_train = X_train / 255
    # train_target = np_utils.to_categorical(train_target, 8)
    print('Train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')        
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id
#---------------------------------------------------------------------------
def load_test(inputSize):
    start_time = time.time()
    path = os.path.join('..', 'input', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(path))
    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img=cv2.resize(cv2.imread(fl),(inputSize,inputSize),cv2.INTER_LINEAR)
        X_test.append(img)
        X_test_id.append(flbase)
    test_data = np.array(X_test, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))
    test_data = test_data.astype('float32')
    test_data = test_data / 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, X_test_id
#---------------------------------------------------------------------------------------------------------
def create_model(inputSize):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1),input_shape=(3, inputSize, inputSize), dim_ordering='th'))#Compute 3777
    model.add(Convolution2D(6, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(6, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(12, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(12, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model
#----------------------------------------------------------------------------
def create_submission(predictions, test_id):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)
#-----Main--------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    # input image dimensions
    imageInputSize=24
    batch_size = 10
    nb_epoch = 18
    #----Initialisation-et-preprocessing-----------
    train_data, train_target, train_id=load_train(imageInputSize)
   #-----A--utiliser--si---loss='categorical_crossentropy'-----------------
    train_target = np_utils.to_categorical(train_target, 8)
   #----------------------------------------------------------------------------------------------------
    test_data, test_id=load_test(imageInputSize)
    model = create_model(imageInputSize)
     #----Training-----------------------------
    print("Training")
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(train_data, train_target, nb_epoch, batch_size)
    # loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32) Pour evaluer la qualité du modele
    print("test")
    #----------------Tests-------------------------
    predictions = model.predict_proba(test_data, batch_size=32)
    print("Submission")
    #------Sauvegarde---predictions---test----
    create_submission(predictions, test_id)
    
    

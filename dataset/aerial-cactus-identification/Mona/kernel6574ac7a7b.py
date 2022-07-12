# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# Any results you write to the current directory are saved as output.

# The following seed-stuff is to get reproducable results with Keras using tensorflow as backend

import numpy as np
np.random.seed(12) #set this seed before importing anything else, in order to get reproducable initial weight initialization

import random as rn
rn.seed(13)

import os
os.environ['PYTHONHASHSEED']=str(0)

#print("is GPU available?")
#print(tf.config.list_physical_devices('GPU'))
#print(tf.config.list_physical_devices('CPU'))

from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import csv
import collections
import pickle
import os
import sys

from keras.models import Sequential, load_model
# import the 'core' layers, which are used in almost any neural network:
from keras.layers import Dense, Dropout, Activation, Flatten
# import the CNN layers, which are used for image analysis:
from keras.layers import Convolution2D, MaxPooling2D
# for transforming the data:from keras.utils import np_utils
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics
from keras.layers.normalization import BatchNormalization
from keras import callbacks
from keras import initializers

from keras import backend as K
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
print(tf.__version__)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/
#
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                        inter_op_parallelism_threads=1)

tf.random.set_seed(14)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


from sklearn import metrics as sklMetrics

from zipfile import ZipFile

cactusPath = "../input/aerial-cactus-identification/"
modelPath = "../input/pretrained-cactusidentification-model/"

dict = {}
with open(cactusPath+"train.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter = ",")
    next(reader, None) # skip first row, as it contains the headers
    for row in reader:
        dict[row[0]] = int(row[1])

sortedDict = collections.OrderedDict(sorted(dict.items()))

y_temp=[]
filenames=[]
for filename, label in sortedDict.items():
    filenames.append(filename)
    y_temp.append(label)

# unzip the train and test data and extact them into the working directory 
# (probably there is a way to work directly with the zip-files, but for now I 
# focus on getting the stuff to work)
with ZipFile(cactusPath+"train.zip","r") as zipObj:
    zipObj.extractall()
#cactusList = zipObjTrain.namelist()
with ZipFile(cactusPath+"test.zip","r") as zipObj:
    zipObj.extractall()
    #testList = zipObjTest.namelist()
    
cactusImageList = [] # will be filled with all the images
testImageList = [] # will be filled with all the images
trainPath="./train/"
testPath="./test/"
cactusList = listdir(trainPath) # returns a list of all the filenames in the path
cactusList.sort()
testList = listdir(testPath) # returns a list of all the filenames in the path

print('cactusList[0]',cactusList[0])

for ind in list(range(0,len(cactusList))):
   image = Image.open(trainPath + cactusList[ind]) #read the image
   cactusImageList.append(image.copy()) # append a copy to the array
   image.close()

# with ZipFile(cactusPath+"train.zip","r") as zipObjTrain:
#     for entry in zipObjTrain.infolist():
#         with zipObjTrain.read(entry) as imageData:
#             fh = StringIO(imageData)
#             image = Image.open(fh)
#             cactusImageList.append(image.copy())
#             #print(img.size, img.mode, len(img.getdata()))
#             image.close()

for ind in list(range(0,len(testList))):
    image = Image.open(testPath + testList[ind]) #read the image
    testImageList.append(image.copy()) # append a copy to the array
    image.close()
    
x_temp = np.stack(cactusImageList) # convert the array of PIL.Images into numpy.ndarray
x_test = np.stack(testImageList) # convert the array of PIL.Images into numpy.ndarray

train_fraction = 0.8
#x_train = x_temp[:int(len(x_temp)*train_fraction)]
#x_val = x_temp[int(len(x_temp)*train_fraction):]

#y_train = y_temp[:int(len(y_temp)*train_fraction)]
#y_val = y_temp[int(len(y_temp)*train_fraction):]

x_train = x_temp[0:1]
x_val = x_temp[1:2]

y_train = y_temp[0:1]
y_val = y_temp[1:2]

hasCactus = y_train.count(1)
hasNoCactus = y_train.count(0)
print("the length of the training data is ", len(x_train))
print("the length of the validation data is ", len(x_val))
print("the fraction of training images with a cactus is: ")
print(float(hasCactus)/float(len(y_train)))
print("the fraction of training images without a cactus is: ")
print(float(hasNoCactus)/float(len(y_train)))

hasCactus = y_val.count(1)
hasNoCactus = y_val.count(0)
print("the fraction of validation images with a cactus is: ")
print(float(hasCactus)/float(len(y_val)))
print("the fraction of validation images without a cactus is: ")
print(float(hasNoCactus)/float(len(y_val)))

print("assign type and divide by 255")
x_train = x_train.astype('float32')
x_train /= 255
x_val  = x_val.astype('float32')
x_val  /= 255
x_test  = x_test.astype('float32')
x_test  /= 255

#print("to categorical:")
#y_train = np_utils.to_categorical(y_train, 2)
#y_val  = np_utils.to_categorical(y_val, 2)

# use imageDataGenerator to create augmented data:
#datagen = ImageDataGenerator(rotation_range=90., zoom_range=0.15, width_shift_range=0.1, height_shift_range=0.1, channel_shift_range=0.2)
datagen = ImageDataGenerator(rotation_range=90., zoom_range=0.2, channel_shift_range=0.2)

# Define my own callback in order to stop training if the validation accuracy has reached a given threshold:
class MyCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') > 0.996:
            self.model.stop_training = True
            print('Stopped training as validation accuracy above threshold')


usePickled = False
nEpochs = 3 ###
if usePickled:
    nEpochs = 0

model_filename = 'cactusIdentification.h5'
history_filename = 'trainHistoryDict'
if usePickled:
    print("load model", modelPath+model_filename)
    model = load_model(modelPath+model_filename)
    print("load history", modelPath+history_filename)
    with open(modelPath+history_filename, 'rb') as file_pi:
        history_dict = pickle.load(file_pi)
else:
    print("define model:")
    model = Sequential()
    model.add(Convolution2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3), kernel_initializer=initializers.RandomUniform(seed=5), bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # reduce the number of nodes by taking the max of every 2x2 patch
    model.add(BatchNormalization())
    model.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer=initializers.RandomUniform(seed=5), bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # reduce the number of nodes by taking the max of every 2x2 patch
    model.add(BatchNormalization())
    model.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer=initializers.RandomUniform(seed=5), bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # reduce the number of nodes by taking the max of every 2x2 patch
    model.add(BatchNormalization())
    #model.add(Dropout(0.3))  # regularize
    model.add(Flatten())
    model.add(Dense(300, activation='relu', kernel_initializer=initializers.RandomUniform(seed=5), bias_initializer='zeros'))
    #model.add(Dropout(0.5))
    model.add(Dense(300, activation='relu', kernel_initializer=initializers.RandomUniform(seed=5), bias_initializer='zeros'))
    #model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid', kernel_initializer=initializers.RandomUniform(seed=5), bias_initializer='zeros'))
    print("")
    print("model summary:")
    print(model.summary())

    print("compile model")
    #model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

if nEpochs != 0:
    print("fit model")
    history = model.fit(x_train,y_train,batch_size=len(x_train),epochs=nEpochs,validation_data=(x_val,y_val),verbose=1) ###
    #history = model.fit(x_train,y_train,batch_size=256,epochs=nEpochs,validation_data=(x_val,y_val),verbose=0) ###
    #history = model.fit_generator(datagen.flow(x_train,y_train,batch_size=32),steps_per_epoch=len(x_train)/32,epochs=nEpochs,validation_data=(x_val,y_val), shuffle=True, callbacks = [MyCallback()], verbose=1)
    model.save(model_filename)
#     with open(history_filename, 'wb') as file_pi:
#         pickle.dump(history.history, file_pi)
    history_dict = history.history

print("evaluate model")
score = model.evaluate(x_val,y_val,verbose=0)
print("The score of the convolutional neural network (CNN) on validation data is:")
print(model.metrics_names)
print(score)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.yscale('log')
ax.plot(history_dict['loss'])
ax.plot(history_dict['val_loss'])
#plt.xticks(np.arange(0, nEpochs, 20))
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid()
plt.legend(['training','val'], loc='upper right')
fig.savefig('trainingHistoryLossLog.png', bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(history_dict['accuracy'])
ax.plot(history_dict['val_accuracy'])
#plt.xticks(np.arange(0, nEpochs, 20))
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid()
plt.legend(['training acc', 'val acc'], loc='lower right')
fig.savefig('trainingHistoryAcc.png', bbox_inches='tight')


# get predictions for val data
y_prob = model.predict(x_val)
y_pred = y_prob.round()
#y_pred = np.zeros((len(y_val),),dtype=np.int)
#for ind,y in enumerate(y_prob) :
#    max_entry = np.argmax(y)
#    y_pred[ind] = max_entry

# get confusion matrix with correct classification
# y_val in row, y_pred in column
#y_val = y_val.argmax(1)
print("Confusion Matrix: \n%s" % sklMetrics.confusion_matrix(y_val,y_pred))


# get predictions for test data
print("get predictions for test data")
y_prob_test = model.predict(x_test)
#y_pred_test = np.zeros((len(y_prob_test),),dtype=np.int)
#for ind,y in enumerate(y_prob_test):
#    max_entry = np.argmax(y)
#    y_pred_test[ind] = max_entry

print("write output file...")
with open("submissionCactus.csv", 'w') as csvfile:
    csvfile.write("id,has_cactus\n")
    for ind in range(len(y_prob_test)):
        #csvfile.write(testList[ind] + "," + str(y_pred_test[ind]) + "\n")
        csvfile.write(testList[ind] + "," + str(y_prob_test[ind]) + "\n")
print("done")
os.system("ls")

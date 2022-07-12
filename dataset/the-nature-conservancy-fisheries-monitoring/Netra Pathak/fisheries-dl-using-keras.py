# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam,Adadelta,Adagrad
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras import backend as K
from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np, os
from PIL import Image
from sklearn.model_selection import train_test_split

# input image dimensions
img_rows, img_cols = 100, 100

# number of channels
img_channels = 3

#  data

trainDir = '../input/train/'  
testDir = '../input/test_stg1/'

categories = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
label = []
imgVectorList = []
labelValue = 0

for folder in categories:
        drc = trainDir+'{}'.format(folder)
        listing = [folder+'/'+im for im in os.listdir(drc)]
        for file in listing:
            im = Image.open(trainDir+file)
            img = np.array(im.resize((img_rows,img_cols)))
            imgVectorList.append(img.flatten())
            label.append(labelValue)
        labelValue += 1
immatrix = np.array(imgVectorList)
label = np.array(label)

# number of output classes
nb_classes = labelValue

immatrix = immatrix.astype('float32')
immatrix -= np.mean(immatrix, axis = 0)
immatrix /= np.std(immatrix, axis = 0)

testData = os.listdir(testDir)
imgVectorList = []
for file in testData:
    imgPath = testDir+file
    im = Image.open(imgPath)
    img = np.array(im.resize((img_rows,img_cols)))
    imgVectorList.append(img.flatten())
X_test = np.array(imgVectorList)

X_test = X_test.astype('float32')

#%%
# Convert class vectors to binary class matrices.
y = to_categorical(label, nb_classes)
# STEP 1: split X and y into training and testing sets
X_train, X_validate, Y_train, Y_validate = train_test_split(immatrix, y, test_size=0.2, random_state=4 ,stratify=y)

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
    X_validate = X_validate.reshape(X_validate.shape[0], img_channels, img_rows, img_cols)
    input_shape = (img_channels, img_rows, img_cols)
    data_format = "channels_first"
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)
    X_validate = X_validate.reshape(X_validate.shape[0], img_rows, img_cols, img_channels)
    input_shape = (img_rows, img_cols, img_channels)
    data_format = "channels_last"


X_train /= 255
X_validate /= 255
X_test /= 255

nb_train_samples=X_train.shape[0]
nb_validation_samples=X_validate.shape[0]

print('X_train shape:', X_train.shape)
print(X_test.shape[0], 'test samples')


#%%
batch_size = 64
epochs = 2
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 3
# convolution kernel size
nb_conv = 3
data_augmentation = True
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# %%
model = Sequential()
model.add(Convolution2D((nb_filters), nb_conv, padding='same', data_format=data_format, kernel_initializer='glorot_normal', input_shape=input_shape, activation='relu'))
model.add(Convolution2D((2*nb_filters), nb_conv, padding='same', kernel_initializer='glorot_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=2))
model.add(Dropout(0.35))

model.add(Convolution2D((3*nb_filters), nb_conv, padding='same', kernel_initializer='glorot_normal', activation='relu'))
model.add(Convolution2D((nb_filters), nb_conv, padding='same', kernel_initializer='glorot_normal',kernel_regularizer=l2(0.01), activation='relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=2))

model.add(Flatten())
model.add(Dense(256, kernel_constraint=maxnorm(3), activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(128, W_constraint=maxnorm(3), kernel_regularizer=l2(0.01), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

print(model.summary())

# categorical_crossentropy is the correct loss metric for multi class classification problem
model.compile(loss='categorical_crossentropy', optimizer='Adagrad', metrics=["accuracy"])


earlystop = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='auto')
# Used to save the model in a filename
# Saves the model only at the epoch which gives the best validation accuracy (because we use 'val_acc')

print('Using real-time data augmentation.')
# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)  # randomly flip images
datagen.fit(X_train)

# Fit the model on the batches generated by datagen.flow().
model.fit_generator(datagen.flow(X_train, Y_train,
                    batch_size=batch_size),
                    steps_per_epoch=(nb_train_samples/batch_size),
                    epochs=epochs,
                    validation_data=(X_validate, Y_validate),
                    validation_steps=(nb_validation_samples/batch_size),
                    callbacks=[earlystop], verbose=1)

pred = model.predict(X_test, verbose=1)
result = pd.DataFrame(pred, columns=categories)
result.insert(0, 'image', testData)
result.to_csv('submissions_stg1.csv', index=False)

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


#https://www.kaggle.com/orhansertkaya/cnn-humpback-whale-identification-with-keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import RMSprop,Adam
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from sklearn.preprocessing import LabelEncoder

import os

train = pd.read_csv('../input/train.csv')
train.head()

ytrain = train['Id']
xtrain = train.drop(labels = ['Id'], axis = 1)
del train

def prepareImages(data, rows, path):
    xdata = np.zeros((rows, 100, 100, 3))
    count = 0
    for fig in data['Image']:
        img = image.load_img('../input/' + path + '/' + fig, target_size = (100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        xdata[count] = x
        count += 1
    return xdata
        
xtrain = prepareImages(xtrain, ytrain.shape[0], 'train')

xtrain = xtrain / 255.0

label_encoder = LabelEncoder()
ytrain = label_encoder.fit_transform(ytrain)
# `to_categorical` converts this into a matrix with as many
# columns as there are classes. The number of rows
# stays the same.
ytrain = to_categorical(ytrain, num_classes = 5005)

model = Sequential()


model.add(Conv2D(filters = 8, kernel_size = (5, 5), padding = 'Same', activation = 'relu', input_shape = (100, 100, 3)))
model.add(Conv2D(filters = 8, kernel_size = (5, 5), padding = 'Same', activation = 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 16, kernel_size = (5, 5), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters = 16, kernel_size = (5, 5), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2), strides = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
model.add(Dropout(0.25))

# fully connected
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(ytrain.shape[1], activation = "softmax"))


optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', 
                                            patience = 3, 
                                            verbose = 1, 
                                            factor = 0.5, 
                                            min_lr = 0.00001)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

epochs = 100  # for better result increase the epochs
batch_size = 1000
model.fit(xtrain, ytrain, epochs = epochs, batch_size = batch_size, verbose = 2, callbacks = [learning_rate_reduction])

test = os.listdir("../input/test/")

col = ['Image']
test_data = pd.DataFrame(test, columns = col)
test_data['Id'] = ''

x_test = prepareImages(test_data, test_data.shape[0], "test")
x_test /= 255

predictions = model.predict(np.array(x_test), verbose = 1)

for i, pred in enumerate(predictions):
    test_data.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
    
test_data.to_csv('submission.csv', index = False)


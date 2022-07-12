# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/train"]).decode("utf8"))
print(check_output(["mkdir", "../input/train/dog ../input/train/cat"]).decode("utf8"))
print(check_output(["mv", "../input/train/dog.* ../input/train/dog/"]).decode("utf8"))
print(check_output(["mv", "../input/train/cat.* ../input/train/cat/"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# import zipfile
# zip_ref = zipfile.ZipFile("/input/train.zip", 'r')
# zip_ref.extractall("/input/train")
# zip_ref = zipfile.ZipFile("../input/test.zip", 'r')
# zip_ref.extractall("/input/test")
# zip_ref.close()

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model
from pathlib import Path


class Cnn():
    def __init__(self):
        load = Path("kuttabilli.h5")
        if load.is_file():
            self.model = load_model('kuttabilli.h5')
            print("Loaded Saved File for retraining!")
        else:
            print("Saved File not found in pwd. Creating new Model")
            # using sequential model
            self.model = Sequential()
            # adding convolution2D layers
            # applies feature detector sliding all over input image
            # gives us feature map which helps making
            # nn run faster and efficiently
            self.model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3),
                                  activation="relu"))
            # to reduceSize of feature maps
            # and reduce no. of node in flatten layer
            self.model.add(MaxPooling2D(pool_size=(3, 3)))
            # 2 conv layer
            self.model.add(Conv2D(64, (3, 3), activation="relu"))
            # to reduceSize of feature maps and
            # reduce no. of node in flatten layer
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            # flatten layer
            self.model.add(Flatten())
            # add dense fc layer
            self.model.add(Dense(units=256, activation='relu'))
            # added a dropout layer to prevent overfitting
            self.model.add(Dropout(rate=0.2))
            # second dense layer ie hidden with half nodes
            self.model.add(Dense(units=128, activation='relu'))
            # added a dropout layer to prevent overfitting
            self.model.add(Dropout(rate=0.1))
            # output layer
            self.model.add(Dense(units=1, activation='sigmoid'))
            # compile model
            self.model.compile(optimizer='adam', metrics=['accuracy'],
                               loss='binary_crossentropy')
            # used adam optimizer and binary_crossentropy due to binary
            # classification in more than 2 class case use
            # categorical_crossentropy
            self.preprocess_image()

    def preprocess_image(self):
        train_set = ImageDataGenerator(
                            rescale=1./255,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True)

        test_set = ImageDataGenerator(rescale=1./255)

        train_set = train_set.flow_from_directory(
                            '../input/train',
                            target_size=(64, 64),
                            batch_size=50,
                            class_mode='binary')

        test_set = test_set.flow_from_directory(
                            '../input/test',
                            target_size=(64, 64),
                            batch_size=10,
                            class_mode='binary')

        self.model.fit_generator(
                            train_set,
                            steps_per_epoch=25000,
                            epochs=15,
                            validation_data=test_set,
                            validation_steps=12500)
        self.model.save('kuttabilli.h5')
        return

    def single_pred(self, path):
        test_img = image.load_img(path, target_size=(64, 64))
        test_img = image.img_to_array(test_img)
        test_img = np.expand_dims(test_img, axis=0)
        result = self.model.predict(test_img)
        if result == 1:
            return 'dog'
        else:
            return 'cat'


cnn = Cnn()
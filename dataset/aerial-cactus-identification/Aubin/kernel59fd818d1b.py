# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:53:52 2019

@author: Aubin
"""

import os
import zipfile
from shutil import copyfile, rmtree# Gestion des fichiers

# Tensorflow pour la classification
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import matplotlib.pyplot as plt# Graphiques
import numpy as np# Manipulation de matrices et de tableaux
import pandas as pd# Manipulation des dataframes

from PIL import Image


def create_dir():
    """ Create dir to classification """
    if not os.path.exists("../working/train/"+'data/cactus'):
        os.makedirs("../working/train/data/cactus")
    if not os.path.exists("../working/train/"+'data/no_cactus'):
        os.makedirs("../working/train/data/no_cactus")
    if not os.path.exists("../working/test"):
        os.makedirs("../working/test")
    with zipfile.ZipFile('../input/aerial-cactus-identification/train.zip', 'r') as zip_ref:
        zip_ref.extractall('../input/aerial-cactus-identification/')
    with zipfile.ZipFile('../input/aerial-cactus-identification/test.zip', 'r') as zip_ref:
        zip_ref.extractall('../input/aerial-cactus-identification/')


def split_cactus():
    """ Get cactus and non cactus images """
    cactus = pd.read_csv('../input/aerial-cactus-identification/train.csv', sep=',')
    image_with_cactus = cactus['id'][cactus['has_cactus'] == 1]
    image_without_cactus = cactus['id'][cactus['has_cactus'] == 0]
    print(image_with_cactus)
    for items in image_with_cactus:
        copyfile('../input/aerial-cactus-identification/train/'+items,
                 '../working/train/data/cactus/'+items)
    for items in image_without_cactus:
        copyfile('../input/aerial-cactus-identification/train/'+items,
                 '../working/train/data/no_cactus/'+items)


# Directories
TRAIN_DATA_DIR = '../working/train/data'
VALIDATION_DATA_DIR = '../working/test'

# Parameters model
NB_TRAIN_SAMPLES = 15000
NV_VALIDATION_SAMPLES = 2500
EPOCHS = 5
BATCH_SIZE = 16
IMG_WIDTH, IMG_HEIGHT = 32, 32 # Images shape
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3) # Le 3 correspond aux 3 valeurs des pixels RGB


def model_(input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE):
    """ Définition du modèle """
    model = Sequential()

    model.add(Conv2D(32, (2, 2), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))#Sigmoid pour classification binaire

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1./32,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./32)

    train_generator = train_datagen.flow_from_directory(TRAIN_DATA_DIR,
                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                        batch_size=batch_size, class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        VALIDATION_DATA_DIR,
        target_size=(32, 32),
        batch_size=batch_size, class_mode='binary')

    model.save('../working/model.h5')
    model.summary()
    return model, train_generator, validation_generator


def save_model(model, filename):
    """ Enregistre le modèle """
    model.save(filename)


def load_model(filename):
    """ Instancie le modèle """
    return tf.keras.models.load_model(filename)


def train_classifier_model(train_generator, validation_generator):
    """ Entraine le modèle """
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=NB_TRAIN_SAMPLES//BATCH_SIZE,
                                  epochs=EPOCHS, validation_data=validation_generator,
                                  validation_steps=NV_VALIDATION_SAMPLES//BATCH_SIZE)
    # Graphique d'évolution de la précision par epoch
    acc = history.history['accuracy']
    epochs_ = range(0, EPOCHS)
    plt.plot(epochs_, acc, label='training accuracy')
    plt.xlabel('no of epochs')
    plt.ylabel('accuracy')
    plt.title("accuracy")
    plt.legend()
    save_model(model, "../working/model.h5")


def predict():
    """ Applique le modèle de classification et renvoie un csv avec l'Id et le résultat """
    image_to_predict = os.listdir('../input/aerial-cactus-identification/test/test')
    result = pd.DataFrame(columns=['id', 'has_cactus'])
    for items in image_to_predict:
        imgpil = Image.open("../input/aerial-cactus-identification/test/test/"+items)
        img = np.array(imgpil)
        img = img.reshape(-1, 32, 32, 3)
        result = result.append({'id': items, 'has_cactus': round(model.predict(img).item(0))},
                               ignore_index=True)
    return result


def replace_int_value(value):
    """ Convertit les valeurs 0.5 en 1 """
    if value == 1:
        return 0.5
    return 0


if __name__ == '__main__':
    create_dir()
    split_cactus()
    model, train_generator, validation_generator = model_()
    train_classifier_model(train_generator, validation_generator)
    #model = load_model("/kaggle/working/model.h5")
    result = predict()
    result['has_cactus'] = result['has_cactus'].apply(lambda x: 1 if x > 0.75 else 0)
    rmtree('../working/train/data/cactus')
    rmtree('../working/train/data/no_cactus')
    result.to_csv('submission.csv', index=False)

# This kernel is completely based on CVxTz's NasNet Mobile kernel: https://www.kaggle.com/CVxTz/cnn-starter-nasnet-mobile
# The only difference is the use of Mobilenet 2 instead of NasNet Mobile 

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from glob import glob
from random import shuffle
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution1D, concatenate, SpatialDropout1D, GlobalMaxPool1D, GlobalAvgPool1D, Embedding, \
    Conv2D, SeparableConv1D, Add, BatchNormalization, Activation, GlobalAveragePooling2D, LeakyReLU, Flatten
from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, \
    Lambda, Multiply, LSTM, Bidirectional, PReLU, MaxPooling1D
from keras.layers.pooling import _GlobalPooling1D
from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy
from keras.models import Model
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df_train = pd.read_csv("../input/train_labels.csv")
id_label_map = {k:v for k,v in zip(df_train.id.values, df_train.label.values)}
df_train.head()

def get_id_from_file_path(file_path):
    return file_path.split(os.path.sep)[-1].replace('.tif', '')
    
    
labeled_files = glob('../input/train/*.tif')
test_files = glob('../input/test/*.tif')

print("labeled_files size :", len(labeled_files))
print("test_files size :", len(test_files))

train, val = train_test_split(labeled_files, test_size=0.1, random_state=101010)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def data_gen(list_files, id_label_map, batch_size, augment=False):
    keras_gen = ImageDataGenerator(
                    rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=True,
                    zoom_range=0.2,
                    shear_range=5)
    while True:
        shuffle(list_files)
        for batch in chunker(list_files, batch_size):
            X = [cv2.imread(x) for x in batch]
            Y = [id_label_map[get_id_from_file_path(x)] for x in batch]
            if augment:
                X = [keras_gen.random_transform(x) for x in X]
            X = [preprocess_input(x.astype(np.float32)) for x in X]
                
            yield np.array(X), np.array(Y)
    
    
def get_model_classif_mobilenet2():
    inputs = Input((96, 96, 3))
    base_model = MobileNetV2(include_top=False, input_shape=(96, 96, 3))#, weights=None
    x = base_model(inputs)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(1, activation="sigmoid", name="3_")(out)
    model = Model(inputs, out)
    model.compile(optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=['acc'])
    model.summary()

    return model
    
model = get_model_classif_mobilenet2()

batch_size=32
h5_path = "model.h5"
checkpoint = ModelCheckpoint(h5_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history = model.fit_generator(
    data_gen(train, id_label_map, batch_size, augment=True),
    validation_data=data_gen(val, id_label_map, batch_size),
    epochs=2, verbose=1,
    callbacks=[checkpoint],
    steps_per_epoch=len(train) // batch_size,
    validation_steps=len(val) // batch_size)
batch_size=64
history = model.fit_generator(
    data_gen(train, id_label_map, batch_size, augment=True),
    validation_data=data_gen(val, id_label_map, batch_size),
    epochs=6, verbose=1,
    callbacks=[checkpoint],
    steps_per_epoch=len(train) // batch_size,
    validation_steps=len(val) // batch_size)
    
model.compile(optimizer=Adam(0.00001), loss=binary_crossentropy, metrics=['acc'])
history = model.fit_generator(
    data_gen(train, id_label_map, batch_size, augment=True),
    validation_data=data_gen(val, id_label_map, batch_size),
    epochs=6, verbose=1,
    callbacks=[checkpoint],
    steps_per_epoch=len(train) // batch_size,
    validation_steps=len(val) // batch_size)

model.load_weights(h5_path)

preds = []
ids = []

for batch in chunker(test_files, batch_size):
    X = [preprocess_input(cv2.imread(x).astype(np.float32)) for x in batch]
    ids_batch = [get_id_from_file_path(x) for x in batch]
    X = np.array(X)
    preds_batch = ((model.predict(X).ravel()*model.predict(X[:, ::-1, :, :]).ravel()*model.predict(X[:, ::-1, ::-1, :]).ravel()*model.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()
    preds += preds_batch
    ids += ids_batch
    
df = pd.DataFrame({'id':ids, 'label':preds})
df.to_csv("baseline_mobilenet_v2.csv", index=False)
df.head()
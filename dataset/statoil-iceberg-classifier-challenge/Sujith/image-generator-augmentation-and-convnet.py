# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

batch_size = 128
input_shape = (75,75,3)
num_classes = 1
epochs = 5

#read the training file
train = pd.read_json("../input/train.json")
train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')
train['inc_angle'] = train['inc_angle'].replace('na,0')
train['inc_angle'] = train['inc_angle'].astype(float).fillna(0.0)
shape1 = train.shape[0]
print(shape1)
x_angle = np.array(train.inc_angle)

#declare arrays to hold training images
arr_band1 = np.zeros((shape1,75,75)) #band_1 data
arr_band2 = np.zeros((shape1,75,75)) #band_2 data
arr_band3 = np.zeros((shape1,75,75)) #band_3 data for image generator
arr_bandall = np.zeros((shape1,75,75,3)) #3 channels
labels = np.zeros(shape1)
angles = np.zeros(shape1)

#move into arrays as 75*75 images --> training
for i in range(train.shape[0]):
    arr_band1[i] = np.reshape(np.array(train.iloc[i, 0]), (75, 75))
    arr_band2[i] = np.reshape(np.array(train.iloc[i, 1]), (75, 75))
    arr_band3[i] = ((arr_band1[i] + arr_band2[i])/2)
    labels[i]=train['is_iceberg'][i]
    angles[i]=x_angle[i]


#read the testing file
test = pd.read_json("../input/test.json")
test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')
test['inc_angle'] = test['inc_angle'].replace('na,0')
test['inc_angle'] = test['inc_angle'].astype(float).fillna(0.0)
shape2 = test.shape[0]
print(shape2)
x_test_angle = np.array(test.inc_angle)

#declare arrays to hold testing images
arr_test_band1 = np.zeros((shape2,75,75)) #band_1 data
arr_test_band2 = np.zeros((shape2,75,75)) #band_2 data
arr_test_band3 = np.zeros((shape2,75,75)) #band_3 data for image generator
arr_test_bandall = np.zeros((shape2,75,75,3)) #3 channels

#move into arrays as 75*75 images --> training
for i in range(test.shape[0]):
    arr_test_band1[i] = np.reshape(np.array(test.iloc[i, 0]), (75, 75))
    arr_test_band2[i] = np.reshape(np.array(test.iloc[i, 1]), (75, 75))
    arr_test_band3[i] = ((arr_test_band1[i] + arr_test_band2[i])/2)

arr_test_bandall = np.concatenate([arr_test_band1[:,:,:,np.newaxis],arr_test_band2[:,:,:,np.newaxis],arr_test_band3[:,:,:,np.newaxis]],
                             axis=-1)
print('test bandall shape')
print(arr_test_bandall.shape)

#---> training starts
#concatenate training data into 3 channels for convolution later by Keras
arr_bandall = np.concatenate([arr_band1[:,:,:,np.newaxis],arr_band2[:,:,:,np.newaxis],arr_band3[:,:,:,np.newaxis]],
                             axis=-1)
print('bandall shape')
print(arr_bandall.shape)

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

datagen.fit(arr_bandall)
x_batches = arr_bandall
y_batches = labels
angle_batches = angles

print('shapes b4 aug')
print(x_batches.shape)
print(y_batches.shape)
print(angle_batches.shape)

genX1 = datagen.flow(arr_bandall, y_batches, batch_size=64, seed=666)
genX2 = datagen.flow(arr_bandall, angle_batches, batch_size=64, seed=666)
epoch_for_gen = 20

for e in range(epoch_for_gen):
    print('Epoch for gen', e)
    X1i = genX1.next()
    X2i = genX2.next()
    x_batches = np.concatenate((x_batches, X1i[0]),axis=0)
    y_batches = np.concatenate((y_batches, X1i[1]),axis=0)
    angle_batches = np.concatenate((angle_batches, X2i[1]),axis=0)

print('shapes after aug')
print(x_batches.shape)
print(y_batches.shape)
print(angle_batches.shape)

x_train, x_test, x_angle_train, x_angle_test, y_train, y_test = train_test_split(x_batches, angle_batches, y_batches)

print('Train', x_train.shape, y_train.shape)
print('Validation', x_test.shape, y_test.shape)

def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]

def get_model():
    bn_model = 0
    p_activation = "relu"
    input_1 = Input(shape=input_shape, name="X_1")
    input_2 = Input(shape=[1], name="angle")

    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)((BatchNormalization(momentum=bn_model))(input_1))
    img_1 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.25)(img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.25)(img_1)
    img_1 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.25)(img_1)
    img_1 = Conv2D(128, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.25)(img_1)
    dense_layer1 = Dropout(0.5)(BatchNormalization(momentum=bn_model)(Dense(32, activation=p_activation)(img_1)))
    dense_out1 = GlobalMaxPooling2D()(dense_layer1)

    img_2 = Conv2D(128, kernel_size=(3, 3), activation=p_activation)((BatchNormalization(momentum=bn_model))(input_1))
    img_2 = MaxPooling2D((2, 2))(img_2)
    img_2 = Dropout(0.2)(img_2)
    dense_layer2 = Dropout(0.5)(BatchNormalization(momentum=bn_model)(Dense(32, activation=p_activation)(img_2)))
    dense_out2 = GlobalMaxPooling2D()(dense_layer2)

    img_concat = (Concatenate()([dense_out1, dense_out2, BatchNormalization(momentum=bn_model)(input_2)]))

    dense_layerf = Dropout(0.5)(BatchNormalization(momentum=bn_model)(Dense(256, activation=p_activation)(img_concat)))
    dense_layerf = Dropout(0.5)(BatchNormalization(momentum=bn_model)(Dense(64, activation=p_activation)(dense_layerf)))
    output = Dense(num_classes, activation="sigmoid")(dense_layerf)

    model = Model([input_1, input_2], output)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

file_path = ".model_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=5)

model = get_model()
model.summary()

model.fit([x_train, x_angle_train], y_train, epochs=25,
          validation_data=([x_test, x_angle_test], y_test), batch_size=32, callbacks=callbacks)

model.load_weights(filepath=file_path)

print("Train evaluate:")
print(model.evaluate([x_train, x_angle_train], y_train, verbose=1, batch_size=200))
print("####################")
print("watch list evaluate:")
print(model.evaluate([x_test, x_angle_test], y_test, verbose=1, batch_size=200))

#----> prediction
prediction = model.predict([arr_test_bandall, x_test_angle], verbose=1, batch_size=200)

submission = pd.DataFrame({'id': test["id"], 'is_iceberg': prediction.reshape((prediction.shape[0]))})
print(submission.head(10))

submission.to_csv("./submission.csv", index=False)


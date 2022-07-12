# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

batch_size = 128
num_classes = 1
epochs = 5
# filters parameters
# window size
winsize = 9
# damping factor for frost
k_value1 = 2.0

def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2
    overall_variance = variance(img)
    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

#read the file
train = pd.read_json('../input/train.json')
train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')
train['inc_angle'] = train['inc_angle'].replace('na,0')
train['inc_angle'] = train['inc_angle'].astype(float).fillna(0.0)
shape1 = train.shape[0]
print(shape1)
input_shape = (75,75,3)

#declare arrays to hold images
arr_band1 = np.zeros((shape1,75,75)) #band_1 data
arr_band2 = np.zeros((shape1,75,75)) #band_2 data
arr_band3 = np.zeros((shape1,75,75)) #band_3 data for image generator
arr_bandall = np.zeros((shape1,75,75,3)) #3 channels
labels = np.zeros(shape1)
flatband = np.zeros(5625)
sin23 = np.sin(((np.pi)/180)*23)
piby180 = ((np.pi)/180)

#move into arrays as 75*75 images
for i in range(train.shape[0]):
    for j in range(5625):
        flatband[j] = train.iloc[i,0][j]
        angle = train['inc_angle'][i]
        if angle == 0:
            continue
        else:
            flatband[j] = ((sin23/np.sin(piby180*angle))*flatband[j]) # standardize to 23 degree incidence angle for all readings
    arr_band1[i] = np.reshape(flatband, (75, 75)) # red band
# Lee filter
    arr_band1[i] = lee_filter(arr_band1[i], 3)
    arr_band2[i] = np.reshape(np.array(train.iloc[i, 1]), (75, 75)) # green band
    arr_band2[i] = lee_filter(arr_band2[i], 3)
    arr_band3[i] = (arr_band1[i] / arr_band2[i]) #take the blue band
    arr_band1[i] = (arr_band1[i] + abs(arr_band1[i].min())) / np.max((arr_band1[i] + abs(arr_band1[i].min()))) #color composite
    arr_band2[i] = (arr_band2[i] + abs(arr_band2[i].min())) / np.max((arr_band2[i] + abs(arr_band2[i].min())))
    arr_band3[i] = (arr_band3[i] + abs(arr_band3[i].min())) / np.max((arr_band3[i] + abs(arr_band3[i].min())))
    labels[i]=train['is_iceberg'][i]

#concatenate into 3 channels for convolution later by Keras
arr_bandall = np.concatenate([arr_band1[:,:,:,np.newaxis],arr_band2[:,:,:,np.newaxis],arr_band3[:,:,:,np.newaxis]],
                             axis=-1)

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

print('shapes b4 aug')
print(x_batches.shape)
print(y_batches.shape)

genX1 = datagen.flow(arr_bandall, y_batches, batch_size=64, seed=666)

epoch_for_gen = 10

for e in range(epoch_for_gen):
    print('Epoch for gen', e)
    X1i = genX1.next()
    x_batches = np.concatenate((x_batches, X1i[0]),axis=0)
    y_batches = np.concatenate((y_batches, X1i[1]),axis=0)

print('shapes after aug')
print(x_batches.shape)
print(y_batches.shape)

#run the testing file
#read the file
test = pd.read_json('../input/test.json')
test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')
test['inc_angle'] = test['inc_angle'].replace('na,0')
test['inc_angle'] = train['inc_angle'].astype(float).fillna(0.0)
shape2 = test.shape[0]
print(shape2)

#declare arrays to hold images
arr_tband1 = np.zeros((shape2,75,75)) #band_1 data
arr_tband2 = np.zeros((shape2,75,75)) #band_2 data
arr_tband3 = np.zeros((shape2,75,75)) #band_3 data for image generator
arr_tbandall = np.zeros((shape2,75,75,3)) #3 channels
flatbandt = np.zeros(5625)

#move into arrays as 75*75 images
for i in range(test.shape[0]):
    for j in range(5625):
        flatbandt[j] = test.iloc[i,0][j]
        tangle = test['inc_angle'][i]
        if tangle == 0:
            continue
        else:
            flatband[j] = ((sin23/np.sin(piby180*tangle))*flatbandt[j]) # standardize to 23 degree incidence angle for all readings
    arr_tband1[i] = np.reshape(flatbandt, (75, 75)) # red band
# lee filter
    arr_tband1[i] = lee_filter(arr_tband1[i], 3)
    arr_tband2[i] = np.reshape(np.array(test.iloc[i, 1]), (75, 75)) # green band
    arr_tband2[i] = lee_filter(arr_tband2[i], 3)
    arr_tband3[i] = (arr_tband1[i] / arr_tband2[i]) #take the blue band
    arr_tband1[i] = (arr_tband1[i] + abs(arr_tband1[i].min())) / np.max((arr_tband1[i] + abs(arr_tband1[i].min()))) #color composite
    arr_tband2[i] = (arr_tband2[i] + abs(arr_tband2[i].min())) / np.max((arr_tband2[i] + abs(arr_tband2[i].min())))
    arr_tband3[i] = (arr_tband3[i] + abs(arr_tband3[i].min())) / np.max((arr_tband3[i] + abs(arr_tband3[i].min())))

#concatenate into 3 channels for convolution later by Keras
arr_tbandall = np.concatenate([arr_tband1[:,:,:,np.newaxis],arr_tband2[:,:,:,np.newaxis],arr_tband3[:,:,:,np.newaxis]],
                             axis=-1)

print('shapes before split')
print(x_batches.shape)
print(y_batches.shape)

x_train, x_test, y_train, y_test = train_test_split(x_batches, y_batches)

print('Train', x_train.shape, y_train.shape)
print('Validation', x_test.shape, y_test.shape)

def get_callbacks(filepath, patience=5):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
#    return msave
    return [es, msave]

def get_model():
    bn_model = 0
    p_activation = "relu"
    input_1 = Input(shape=input_shape, name="X_1")

    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)((BatchNormalization(momentum=bn_model))(input_1))
    img_1 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.25)(img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1_in = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.25)(img_1_in)
    img_1 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.25)(img_1)
    img_1 = Conv2D(128, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.25)(img_1)
    img_1 = GlobalMaxPooling2D()(img_1)

    img_2 = Conv2D(128, kernel_size=(3, 3), activation=p_activation)((BatchNormalization(momentum=bn_model))(input_1))
    img_2 = MaxPooling2D((2, 2))(img_2)
    img_2 = Dropout(0.2)(img_2)
    img_2 = GlobalMaxPooling2D()(img_2)
    
    img_3 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)(img_1_in)
    img_3 = MaxPooling2D((2, 2))(img_3)
    img_3 = Dropout(0.2)(img_3)
    img_3 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)(img_3)
    img_3 = MaxPooling2D((2, 2))(img_3)
    img_3 = Dropout(0.25)(img_3)
    img_3 = Conv2D(128, kernel_size=(3, 3), activation=p_activation, padding='same')(img_3)
    img_3 = MaxPooling2D((2, 2))(img_3)
    img_3 = Dropout(0.25)(img_3)
    img_3 = GlobalMaxPooling2D()(img_3)

    img_concat = (Concatenate()([img_1, img_2, img_3]))

    dense_layer = Dropout(0.5)(BatchNormalization(momentum=bn_model)(Dense(256, activation=p_activation)(img_concat)))
    dense_layer = Dropout(0.5)(BatchNormalization(momentum=bn_model)(Dense(64, activation=p_activation)(dense_layer)))
    output = Dense(num_classes, activation="sigmoid")(dense_layer)

    model = Model(input_1, output)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

file_path = '.model_weights3.hdf5'
callbacks = get_callbacks(filepath=file_path, patience=5)

print('defining model.....')
model = get_model()
model.summary()
print('fitting....')
model.fit(x_train, y_train, epochs=25,
          validation_data=(x_test, y_test), batch_size=128, callbacks=callbacks)

model.load_weights(filepath=file_path)

print("Train evaluate:")
print(model.evaluate(x_train, y_train, verbose=1, batch_size=200))
print("####################")
print("watch list evaluate:")
print(model.evaluate(x_test, y_test, verbose=1, batch_size=200))
# Any results you write to the current directory are saved as output.
predicted_test = model.predict(arr_tbandall)
submission = pd.DataFrame()
submission['id'] = test['id']
submission['is_iceberg'] = predicted_test.reshape((predicted_test.shape[0]))
submission.to_csv('sub3.csv', index=False)

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from glob import glob 
from skimage.io import imread
import gc

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import shutil


import os
import pandas as pd
import numpy as np
import PIL
import matplotlib
import matplotlib.pyplot as plt
from keras.models import load_model,save_model
from keras.callbacks import ModelCheckpoint
from matplotlib.patches import Rectangle
import os
from scipy.misc import imsave
from tqdm import tqdm

import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.applications import xception
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam,Adagrad
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU
from math import sqrt
from keras.callbacks import History 
from keras.optimizers import Adam, SGD
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.applications import xception
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, merge, GlobalAveragePooling2D, MaxPool2D, GlobalMaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,Input, Concatenate
from keras.layers import BatchNormalization
from keras.models import Model
from keras.activations import relu
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import ELU
import keras.backend as K
from math import sqrt
from keras.callbacks import History 

from keras.applications import ResNet50, VGG19, NASNetMobile,DenseNet169
import gc


trainPath = '../input/train/'
df = pd.read_csv("../input/train_labels.csv")

df = df[df['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']
df = df[df['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']

def read_img(filepath, size,grayscale=False):
    
    if grayscale:
        img = image.load_img((filepath), target_size=size,grayscale=True)
        img = image.img_to_array(img,data_format='channels_last')
    else:
        img = image.load_img((filepath), target_size=size)
        img = image.img_to_array(img,data_format='channels_last')
    return img


lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=1e-5, patience=3, verbose=1)

modelID=0
checkPoint = ModelCheckpoint(filepath='./model_resnet_{epoch:02d}_{val_loss:.4f}_{val_acc:4f}.hdf5',verbose=1,save_best_only=True,mode=min)
bestCheckPoint = ModelCheckpoint(filepath='./bestModel.hdf5',verbose=0,save_best_only=True,mode=min)

datagen = ImageDataGenerator(preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x,
                            horizontal_flip=True,
                            vertical_flip=True,
                            validation_split=0.1)
                            
trainGen=datagen.flow_from_dataframe(df, trainPath, x_col='id',y_col='label',has_ext=False,target_size=(96,96),subset='training', class_mode='binary',shuffle=True)
valGen = datagen.flow_from_dataframe(df, trainPath, x_col='id',y_col='label',has_ext=False,target_size=(96,96),subset='validation', class_mode='binary',shuffle=True)

def conv_layer(feature_batch, feature_map, kernel_size=(3, 3),strides=(1,1), zp_flag=False):
    if zp_flag:
        zp = ZeroPadding2D((1,1))(feature_batch)
    else:
        zp = feature_batch
    conv = Conv2D(filters=feature_map, kernel_size=kernel_size, strides=strides)(zp)
    bn = BatchNormalization(axis=3)(conv)
    act = LeakyReLU(1/10)(bn)
    return act
    
    
model = DenseNet169(include_top = False, input_shape=(96,96,3))

# for layer in model.layers:
#     layer.trainable = False


x = model.output

out1 = GlobalMaxPooling2D()(x)
out2 = GlobalAveragePooling2D()(x)
# out3 = Flatten()(x)
out = Concatenate(axis=-1)([out1, out2])
out = Dropout(0.4)(out)
out = Dense(256, activation="relu")(out)
predictions = Dense(1, activation="sigmoid")(out)




fullModel = Model(inputs=model.input, outputs=predictions)

adam = Adam(lr=0.00001)

fullModel.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])



fullModel.fit_generator(trainGen,epochs=14,steps_per_epoch=2000,validation_data=valGen, validation_steps=100,callbacks=[lr_reduce,checkPoint,bestCheckPoint])

fullModel.load_weights('./bestModel.hdf5')


base_test_dir = '../input/test/'
test_files = glob(os.path.join(base_test_dir,'*.tif'))
submission = pd.DataFrame()
file_batch = 5000
max_idx = len(test_files)
for idx in range(0, max_idx, file_batch):
    print("Indexes: %i - %i"%(idx, idx+file_batch))
    test_df = pd.DataFrame({'path': test_files[idx:idx+file_batch]})
    test_df['id'] = test_df.path.map(lambda x: x.split('/')[3].split(".")[0])
    test_df['image'] = test_df['path'].map(imread)
    K_test = np.stack(test_df["image"].values)
    K_test = (K_test - K_test.mean()) / K_test.std()
    predictions = fullModel.predict(K_test)
    test_df['label'] = predictions
    submission = pd.concat([submission, test_df[["id", "label"]]])
submission.head()

submission.to_csv("submission.csv", index = False, header = True)

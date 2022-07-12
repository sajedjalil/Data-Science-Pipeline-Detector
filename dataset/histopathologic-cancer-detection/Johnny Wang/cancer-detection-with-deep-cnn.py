# Cancer detection
# Johnny Wang - johnny.wang@live.ca
# CNN based on Ciresan et al. https://link.springer.com/content/pdf/10.1007%2F978-3-642-40763-5_51.pdf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import random
import cv2
import tensorflow as tf
from glob import glob
from random import shuffle


from keras import layers
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.losses import  binary_crossentropy
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import glorot_uniform
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn import metrics
#from sklearn.metrics import roc_auc_score
from tflearn.objectives import roc_auc_score


# Parameters
learning_rate = 0.001
epochs = 3
batch_size = 16
test_size = 0.10
seed = 101101101

np.random.seed(seed)

# AUGMENTATION VARIABLES
ORIGINAL_SIZE = 96      # original size of the images - do not change
CROP_SIZE = 64          # final size after crop
RANDOM_ROTATION = 180   # range (0-180), 180 allows all rotation variations, 0=no change
RANDOM_SHIFT = 4        # center crop shift in x and y axes, 0=no change
RANDOM_BRIGHTNESS = 10   # range (0-100), 0=no change
RANDOM_CONTRAST = 10     # range (0-100), 0=no change


# Load data into Python
df_train = pd.read_csv("../input/train_labels.csv")
id_label_map = {k:v for k,v in zip(df_train.id.values, df_train.label.values)}

def get_id_from_file_path(file_path):
    return file_path.split(os.path.sep)[-1].replace('.tif', '')
    
labeled_files = glob('../input/train/*.tif')
test_files = glob('../input/test/*.tif')
train, val = train_test_split(labeled_files, test_size=0.1, random_state=seed)

def readCroppedImage(path,augment):
    # OpenCV reads the image in bgr format by default
    bgr_img = cv2.imread(path)
    # We flip it to rgb for visualization purposes
    b,g,r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])
    x = 0
    y = 0
    if augment:
        #random rotation
        rotation = random.randint(-RANDOM_ROTATION,RANDOM_ROTATION)  
        M = cv2.getRotationMatrix2D((48,48),rotation,1)
        rgb_img = cv2.warpAffine(rgb_img,M,(96,96))
        #random x,y-shift
        x = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT)
        y = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT)

        # Random flip
        flip_hor = bool(random.getrandbits(1))
        flip_ver = bool(random.getrandbits(1))
        if(flip_hor):
            rgb_img = rgb_img[:, ::-1]
        if(flip_ver):
            rgb_img = rgb_img[::-1, :]
        # Random brightness
        br = random.randint(-RANDOM_BRIGHTNESS, RANDOM_BRIGHTNESS) / 100.
        rgb_img = rgb_img + br
        # Random contrast
        cr = 1.0 + random.randint(-RANDOM_CONTRAST, RANDOM_CONTRAST) / 100.
        rgb_img = rgb_img * cr
    
    # crop to center and normalize to 0-1 range
    start_crop = (ORIGINAL_SIZE - CROP_SIZE) // 2
    end_crop = start_crop + CROP_SIZE
    rgb_img = rgb_img[(start_crop + x):(end_crop + x), (start_crop + y):(end_crop + y)] / 255
    
    # clip values to 0-1 range
    rgb_img = np.clip(rgb_img, 0, 1.0)
    return rgb_img

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def data_gen(list_files, id_label_map, batch_size, augment=False):
    while True:
        shuffle(list_files)
        for batch in chunker(list_files, batch_size):
            X = [readCroppedImage(x,augment) for x in batch]
            Y = [id_label_map[get_id_from_file_path(x)] for x in batch]
            yield np.array(X), np.array(Y)
            
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def identity_block(X, f, filters, stage, block):

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X
    
def convolutional_block(X, f, filters, stage, block, s = 2):
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'same', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X



def ResNet50_model(input_shape = (CROP_SIZE, CROP_SIZE, 3), classes = 1):
    inputs = Input(input_shape)

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128,128,512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128,128,512], stage = 3, block='b')
    X = identity_block(X, 3, [128,128,512], stage = 3, block='c')
    X = identity_block(X, 3, [128,128,512], stage = 3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256,256,1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256,256,1024], stage = 4, block='b')
    X = identity_block(X, 3, [256,256,1024], stage = 4, block='c')
    X = identity_block(X, 3, [256,256,1024], stage = 4, block='d')
    X = identity_block(X, 3, [256,256,1024], stage = 4, block='e')
    X = identity_block(X, 3, [256,256,1024], stage = 4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512,512,2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512,512,2048], stage = 5, block='b')
    X = identity_block(X, 3, [512,512,2048], stage = 5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), name = 'avg_pool')(X)
    

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')
    #model.compile(Adam(learning_rate), loss=binary_crossentropy, metrics=['acc',auc])
    model.compile(Adam(learning_rate), loss=roc_auc_score, metrics=['acc',auc])
    return model

model = ResNet50_model()


h5_path = "model.res50"
checkpoint = ModelCheckpoint(h5_path, monitor='val_auc', verbose=1, save_best_only=True, mode='max')
earlystopper = EarlyStopping(monitor='auc', patience=5, verbose=1)

    
batch_size = 32
history = model.fit_generator(
    data_gen(train, id_label_map, batch_size, augment=False),
    validation_data=data_gen(val, id_label_map, batch_size),
    epochs=epochs, verbose=1,
    callbacks=[checkpoint,earlystopper],
    steps_per_epoch=len(train) // batch_size,
    validation_steps=len(val) // batch_size)

batch_size = 16
history = model.fit_generator(
    data_gen(train, id_label_map, batch_size, augment=True),
    validation_data=data_gen(val, id_label_map, batch_size),
    epochs=epochs, verbose=1,
    callbacks=[checkpoint,earlystopper],
    steps_per_epoch=len(train) // batch_size,
    validation_steps=len(val) // batch_size)

model.load_weights(h5_path)

preds = []
ids = []
for batch in chunker(test_files, batch_size):
    X = [readCroppedImage(x,False) for x in batch]
    ids_batch = [get_id_from_file_path(x) for x in batch]
    X = np.array(X)
    preds_batch = ((model.predict(X).ravel()*model.predict(X[:, ::-1, :, :]).ravel()*model.predict(X[:, ::-1, ::-1, :]).ravel()*model.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()
    preds += preds_batch
    ids += ids_batch
df = pd.DataFrame({'id':ids, 'label':preds})
df.to_csv("submit.csv", index=False)
df.head()



URL_STEM = 'http://andy.harless.us/rsnaweights/'

weight_files = ['model_dense_v5_a1_f' + str(i) + '.h5' for i in range(5)]

BATCHSIZE = 8
CHANNELS = 64
IMAGE_SIZE = 512
NBLOCK = 4
DEPTH = 5
MOMENTUM = 0.9

import os
import csv
import random
import pydicom
import numpy as np
import pandas as pd
from skimage import measure
from skimage.transform import resize
from keras.models import load_model

import tensorflow as tf
from tensorflow import keras
from subprocess import call
import hashlib



# Data generator

class generator(keras.utils.Sequence):
    
    def __init__(self, folder, filenames, pneumonia_locations=None, batch_size=BATCHSIZE, 
                 image_size=IMAGE_SIZE, shuffle=True, augment=False, predict=False):
        self.folder = folder
        self.filenames = filenames
        self.pneumonia_locations = pneumonia_locations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()
        
    def __load__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # default negative
        target = 0
        # get filename without extension
        filename = filename.split('.')[0]
        # if image contains pneumonia
        if filename in pneumonia_locations:
            target = 1
        # resize both image and mask
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        # if augment then horizontal flip half the time
        if self.augment and random.random() > 0.5:
            img = np.fliplr(img)
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        return img, target
    
    def __loadpredict__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # resize image
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        return img
        
    def __getitem__(self, index):
        # select batch
        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        # predict mode: return images and filenames
        if self.predict:
            # load files
            imgs = [self.__loadpredict__(filename) for filename in filenames]
            # create numpy batch
            imgs = np.array(imgs)
            return imgs, filenames
        # train mode: return images and masks
        else:
            # load files
            items = [self.__load__(filename) for filename in filenames]
            # unzip images and masks
            imgs, targets = zip(*items)
            # create numpy batch
            imgs = np.array(imgs)
            targets = np.array(targets)
            return imgs, targets
        
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)
        
    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.filenames) / self.batch_size))
        else:
            # return full batches only
            return int(len(self.filenames) / self.batch_size)
            
            
            
# Network

def convlayer(channels, inputs, size=3, padding='same'):
    x = keras.layers.BatchNormalization(momentum=MOMENTUM)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, size, padding=padding, use_bias=False)(x)
    return x

def just_downsample(inputs, pool=2):
    x = keras.layers.BatchNormalization(momentum=MOMENTUM)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.MaxPool2D(pool)(x)
    return x

def convblock(inputs, channels1, channels2):
    x = convlayer(channels1, inputs)
    x = convlayer(channels2, x)
    x = keras.layers.Concatenate()([inputs, x])
    return x

def denseblock(inputs, nblocks=6, channels1=128, channels2=32, do=.2):
    x = inputs
    for i in range(nblocks):
        x = convblock(x, channels1, channels2)
    x = keras.layers.SpatialDropout2D(do)(x) # .2
    return x

def transition(inputs, channels, pool=2):
    x = convlayer(channels, inputs)
    x = keras.layers.AveragePooling2D(pool)(x)
    return x
    
def create_network(input_size, channels=64, channels2=32, n_blocks=NBLOCK, depth=DEPTH):
    # input
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = keras.layers.Conv2D(channels, 3, padding='same', strides=2, use_bias=False)(inputs)
    x = just_downsample(x)

    # densenet blocks
    basedo = .05
    do = basedo
    nchan = channels
    for d in range(depth-1):
        x = denseblock(x, do=do)
        nchan = ( nchan + n_blocks*channels2 ) // 2
        x = transition(x, nchan)
        do = do + basedo/2
    x = denseblock(x, do=do)

    # output
    x = convlayer(channels, x)
    x = keras.layers.BatchNormalization(momentum=MOMENTUM)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Dropout(.5)(x)
    output = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=output)
    return model
    
    
    
model = create_network(input_size=IMAGE_SIZE, channels=CHANNELS, n_blocks=NBLOCK, depth=DEPTH)


with open('md5sums.log', 'w') as md5file:

    for wf in weight_files:
    
        file_url = URL_STEM + wf
        call( ['wget', '-q', file_url] )
        md5file.write( hashlib.md5(open(wf,'rb').read()).hexdigest() + ' ' + wf + '\n' )

        model.load_weights(wf)
        
        
        # Predict test images
        
        folder = '/kaggle/input/stage_2_test_images'
        test_filenames = filenames = os.listdir(folder)
        print('n test samples:', len(test_filenames))
        
        # create test generator with predict flag set to True
        test_gen = generator(folder, test_filenames, None, batch_size=25, 
                             image_size=IMAGE_SIZE, shuffle=False, predict=True)
                             
        count = 0
        test_predictions = []
        test_patient_ids = []
        for imgs, filenames in test_gen:
            # predict batch of images
            preds = model.predict(imgs)
            # loop through batch
            for pred, filename in zip(preds, filenames):
                count = count + 1
                patient = filename.split('.')[0]
                test_predictions = test_predictions + list(pred)
                test_patient_ids.append( patient )
        
            # stop if we've got them all
            if count >= len(test_filenames):
                break
            
        test_df = pd.DataFrame({'patientId':test_patient_ids, 'predicted':test_predictions})
        
        version = wf.split('.')[-2].split('_')[-1]
        submission_fp = f'submission_adense_{version}.csv'
        
        test_df.to_csv(submission_fp, index=False)


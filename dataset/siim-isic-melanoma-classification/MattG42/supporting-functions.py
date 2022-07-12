""" Imports """
import numpy as np
import pandas as pd

import re

import keras
import tensorflow as tf
import random

from keras.preprocessing import image
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten,BatchNormalization,Activation
from keras.preprocessing.image import ImageDataGenerator



from keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.keras import backend as K

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

from kaggle_datasets import KaggleDatasets



""" Setup """

# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE


# Configuration
BATCH_SIZE = 32 * 8 # kind of a hack. We should have access to 8 TPUs
IMAGE_SIZE = [512, 512]
NUM_CLASSES = 1





""" Supporting Functions """

def process_image(raw_image):
    
    image = raw_image
    
    if(IMAGE_SIZE != [1024, 1024]):
        image = tf.image.resize(image, IMAGE_SIZE)
    
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    
    return image

def data_augment(image):
    
    rand = random.randint(0,4)
    
    if rand == 0:
        image = tf.image.random_flip_left_right(image)
        
    elif rand == 1:
        image = tf.image.random_flip_up_down(image)
        
    elif rand == 2:
        image = tf.image.random_contrast(image, 0, 2, seed=10)
        
    elif rand == 3:
        image = tf.image.random_brightness(image, 1, seed=10)
        
    elif rand == 4:
        image = tf.image.random_saturation(image, 0, 2, seed=10)
    
    return image 
    

def read_train_record(data):
  
    features = {
    # tf.string = byte string (not text string)
    "image": tf.io.FixedLenFeature([], tf.string), # shape [] means scalar, here, a single byte string
    "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar, i.e. a single item
    }

    # decode the TFRecord
    tf_record = tf.io.parse_example(data, features)

    # Typical code for decoding compressed images
    image = tf.image.decode_jpeg(tf_record['image'], channels=3)
    
    image = process_image(image)
    
    image = data_augment(image)
    
    target = tf_record['target']

    return image, target

def read_val_record(data):
  
    features = {
    # tf.string = byte string (not text string)
    "image": tf.io.FixedLenFeature([], tf.string), # shape [] means scalar, here, a single byte string
    "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar, i.e. a single item
    }

    # decode the TFRecord
    tf_record = tf.io.parse_example(data, features)

    # Typical code for decoding compressed images
    image = tf.image.decode_jpeg(tf_record['image'], channels=3)
    
    image = process_image(image)
    
    target = tf_record['target']

    return image, target


def get_training_dataset(files):
    
    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_train_record, num_parallel_calls=AUTO)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    
    return dataset

def get_val_dataset(files):
    
    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.repeat()
    dataset = dataset.map(read_test_record, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    
    return dataset


def get_sizes():
    
    return IMAGE_SIZE, BATCH_SIZE
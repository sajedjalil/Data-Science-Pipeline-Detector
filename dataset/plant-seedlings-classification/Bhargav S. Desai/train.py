"""
This is a training script. 

It trains on the Plant Seedling data with VGG16 network and saves it on disk. Expected accuracy ~91%. 
"""


import numpy as np
import itertools
import pandas as pd
import os

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


from utils import DataLoader, Utilities


# Set some global variables
train_dir = "../input/plant-seedlings-classification/train/"
test_dir = "../input/plant-seedlings-classification/test/"
save_dir = "/kaggle/working/plant-seedlings-classification/train"
target_size = (224, 224)



# Initialize DataLoader
dataloader = DataLoader(train_dir = train_dir, test_dir = test_dir, save_dir = save_dir, target_size = target_size, segmentation = True)


# Initialize Utilities
utils = Utilities(train_dir, save_dir)


# Balance the dataset
dataloader.balance_dataset()


# Load generators for the data using the DataLoader Class
train_generator, val_generator = dataloader.load_for_train(model = "vgg16")


# Define callbacks
model_save_path = '/kaggle/working/model_vgg16.h5'
checkpoint = ModelCheckpoint(filepath=model_save_path, monitor='val_loss', mode='min', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.00000001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min', restore_best_weights=True)


# Configure model for transfer learning
base_model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
x = base_model.output
x = Dropout(0.5)(x)
predictions = Dense(12, activation='softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)


# Freeze the earlier layers
for layer in model.layers[:-11]:
    layer.trainable = False
    
    
# Compile the model    
model.compile(Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])



# Train the model
model.fit_generator(train_generator,
                      steps_per_epoch = 196,
                      validation_data = val_generator,
                      validation_steps = 48,
                      epochs = 32,
                      verbose = 1,
                      callbacks = [reduce_lr, checkpoint, early_stop])
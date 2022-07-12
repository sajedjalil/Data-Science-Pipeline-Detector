# To use less memory during training can use "flow_from_directory()"
# Manually create training and validation folders (I took first 40 out of each training and put in corresponding validation sub-folder)
# Structure (~10% split):
# 1. validate_path/class_1/(first 40 from class_1 in training set)
#   "            "/class_2/(first 40 from class_2 in training set)
#   ...
#   "            "/class_12/(first 40 from class_12 in training set)
# Result is 480 validation images 

# 2. train_path/class_1/(without first 40 from class_1 in training set)
#   "         "/class_2/(without first 40 from class_2 in training set)
#   ...
#   "         "/class_12/(without first 40 from class_12 in training set)
# Result is 4270 training images

# 3. test_path/test/(all 794 test set images)
# Result is 794 test images

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, cv2
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.applications import *

train_path = 'train_path'
validate_path = 'validate_path'
test_path = 'test_path'

df_test = pd.read_csv('sample_submission.csv')

dim = 224 # Must choose one of 128, 160, 192, 224 if using imagenet weights
epochs = 100
learning_rate = 0.001
batch_size = 12 # For those with low VRAM (~2GB)
# Train/validation split
trn_num = 4270
val_num = 480

weights = os.path.join('', 'weights.h5')

datagen = ImageDataGenerator( horizontal_flip=True, 
		                      vertical_flip=True,
		                      width_shift_range=0.1,
		                      height_shift_range=0.1)

train_generator = datagen.flow_from_directory(
        train_path,  
        target_size=(dim, dim), 
        batch_size=batch_size,
        class_mode='categorical') 

validate_generator = datagen.flow_from_directory(
        validate_path,
        target_size=(dim, dim),
        batch_size=batch_size,
        class_mode='categorical')

callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0), ModelCheckpoint(weights, monitor='val_loss', save_best_only=True, verbose=0), ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)]

# MobileNet is smaller in size and depth (https://keras.io/applications/) - can train on larger dimension inputs.
base_model = MobileNet(input_shape=(dim, dim, 3), include_top=False, weights='imagenet', pooling='avg')
x = base_model.output
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(12, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

#for layer in base_model.layers:
#    layer.trainable = False

#print(model.summary())

model.fit_generator(train_generator,
                    steps_per_epoch=trn_num/batch_size, 
                    validation_data=validate_generator, 
                    validation_steps=val_num/batch_size,
                    callbacks=callbacks,
                    epochs=epochs, 
                    verbose=1)
                    
# ------ TESTING ------
label_map = validate_generator.class_indices
print(label_map)

test_datagen = ImageDataGenerator() # No augmentation on test dataset
generator = test_datagen.flow_from_directory(
        test_path,  
        target_size=(dim, dim), 
        batch_size=batch_size,
        class_mode=None, # No labels for test dataset
        shuffle=False) # Do not shuffle data to keep labels in same order

if os.path.isfile(weights):
    model.load_weights(weights)

p_test = model.predict_generator(generator, verbose=1)

preds = []
for i in range(len(p_test)):
    pos = np.argmax(p_test[i])
    preds.append(list(label_map.keys())[list(label_map.values()).index(pos)])
    
df_test['species'] = preds
df_test.to_csv('submission.csv', index=False)
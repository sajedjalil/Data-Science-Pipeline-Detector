import pandas as pd
import numpy as np
import os,cv2
from time import time
from keras.preprocessing import image
from keras import optimizers
from keras import layers,models
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import DenseNet121, DenseNet169, DenseNet201
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# No tensorboard for kaggle kernel
# from keras.callbacks import TensorBoard

from keras import backend as K

"""
file directory setup:

            ariel_cactus_identification
                        |
    test    train   train.csv   sample_submission.csv   logs
     |        |                                          |
*.jpg (4000)  *.jpg (17500)                         <script run timestamp (ts)>
                                                         |
                                                    events.out.tfevents.<ts>
"""


# set up project dirs
project_dir = "../input"
train_dir = f"{project_dir}/train/train"
test_dir = f"{project_dir}/test/test"
# load in labels df 
train_df = pd.read_csv(f'{project_dir}/train.csv') # id (references filename), label
test_df = pd.read_csv(f'{project_dir}/sample_submission.csv') 
# label arrays
train_id = train_df['id']
labels = train_df['has_cactus']
test_id = test_df['id']

# set up tensorboard
# tensorboard = TensorBoard(log_dir=f"{project_dir}/logs/{time()}")

# Set up images
def get_images(ids, filepath):
    arr = []
    for img_id in ids:
        img = plt.imread(f"{filepath}/{img_id}")
        arr.append(img)
    
    arr = np.array(arr).astype('float32')
    arr = arr / 255
    return arr

# split the ids and lables into train/val sets 
x_train, x_val, y_train, y_val = train_test_split(train_id, labels, test_size=0.2)

# pull the images associated with each id into an array
x_train = get_images(ids=x_train, filepath=train_dir) # returns an ndarray (14000, 32, 32, 3)
x_val = get_images(ids=x_val, filepath=train_dir) # returns an ndarray (3500, 32, 32, 3)
test = get_images(ids=test_id, filepath=test_dir) # returns an ndarray (4000, 32, 32, 3)

img_dim = x_train.shape[1:] # dimension of a single image

batch_size = 64
epochs = 30
steps = x_train.shape[0] // batch_size

inputs = Input(shape=img_dim)
densenet121 = DenseNet121(weights='imagenet', include_top=False)(inputs)
flat1 = Flatten()(densenet121)
dense1 = Dense(units=256, use_bias=True)(flat1)
batchnorm1 = BatchNormalization()(dense1)
act1 = Activation(activation='relu')(batchnorm1)
drop1 = Dropout(rate=0.5)(act1)
out = Dense(units=1, activation='sigmoid')(drop1)

model = Model(inputs=inputs, outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy')
# callback that reduces the learning rate, if vali
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=2, mode='max')

img_aug = ImageDataGenerator(rotation_range=20, vertical_flip=True, horizontal_flip=True)
img_aug.fit(x_train)

model.fit_generator(img_aug.flow(x_train, y_train, batch_size=batch_size), 
                    steps_per_epoch=steps, epochs=epochs, 
                    validation_data=(x_val, y_val), callbacks=[reduce_lr], # , tensorboard
                    verbose=2)

test_pred = model.predict(test, verbose=2)
test_df['has_cactus'] = test_pred
test_df.to_csv('submission.csv', index=False)
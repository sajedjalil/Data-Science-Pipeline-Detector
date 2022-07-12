import skimage
from skimage import transform, color
import skimage.io as io
import numpy as np # linear algebra
from numpy import array
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
from tensorflow import keras
import csv
#from scikit-image import *

def grayscale_image_to_color(img):
    return skimage.color.gray2rgb(img)

def fetch_images(target):
     return [os.path.join('../input/'+target, f)
                for f in os.listdir('../input/'+target)
                if f.endswith(".jpg")]
    
labels = []
images = []
images_predict= []
class_names = []
df_train = pd.read_csv('../input/train.csv')
class_names = list(set(df_train['Id'].tolist()))
train_images = fetch_images('train')
for f in train_images:
    img = io.imread(f)
    if len(img.shape) == 2:
        img = grayscale_image_to_color(img)
    img = transform.resize(img, (56, 56))
    images.append(img)
    img_name = f.split('/')[-1]
    labels.append(class_names.index(df_train.loc[df_train['Image'] == img_name, 'Id'].iloc[0]))
images = array(images)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(56, 56, 3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(5005, activation=tf.nn.softmax)])
model.compile(optimizer=tf.train.AdamOptimizer(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
model.fit(images, labels, epochs=5)
test_images = fetch_images('test')
for f in test_images:
        img = io.imread(f)
        if len(img.shape) == 2:
            img = grayscale_image_to_color(img)
        img = transform.resize(img, (56, 56))
        images_predict.append(img)
images_predict = array(images_predict)
predictions = model.predict(images_predict)
np.savetxt("predictions.csv", predictions, delimiter=",")


"""
Submission for: https://www.kaggle.com/xhlulu/cassava-train-efficientnet-on-tpu-in-100-lines
"""
import os
os.system('pip install /kaggle/input/kerasapplications -q')
os.system('pip install /kaggle/input/efficientnet-keras-source-code/ -q --no-deps')

import efficientnet.tfkeras as efn
import numpy as np
import pandas as pd
from kaggle_datasets import KaggleDatasets
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import load_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Helper functions
def decode_image(path, label=None, target_size=(512, 512)):
    img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, target_size)
    
    return img if label is None else img, label

def data_augment(img, label=None):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    
    return img if label is None else img, label

strategy = tf.distribute.get_strategy()
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16

load_dir = "/kaggle/input/cassava-leaf-disease-classification/"
sub_df = pd.read_csv(load_dir + 'sample_submission.csv')
sub_df['paths'] = load_dir + "/test_images/" + sub_df.image_id


test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(sub_df.paths.values)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE).prefetch(AUTO))

with strategy.scope():
    model = load_model(
        '/kaggle/input/'
        'cassava-train-efficientnet-on-tpu-in-100-lines/'
        'model.h5')
    model.summary()

preds = model.predict(test_dataset, verbose=1)
sub_df['label'] = preds.argmax(axis=1)
sub_df.drop(columns='paths').to_csv('submission.csv', index=False)
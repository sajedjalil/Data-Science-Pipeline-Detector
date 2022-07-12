import matplotlib.image as mpimg
import numpy as np
import os
import keras
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa


def import_data(directory):
    file_list = os.listdir(directory)[:300]
    images = [mpimg.imread(directory + '/' + i).copy().astype(np.float32)/127.5 - 1 for i in file_list]

    images = tf.constant(np.array(images), dtype=tf.float32)  # X is a np.array

    return images


def train_test(dataset, train_size):
    train_dim = int(len(dataset) * train_size)

    train, test = dataset[0:train_dim], dataset[train_dim:]

    return train, test


def downsample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    result.add(layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result


def Generator():
    inputs = layers.Input(shape=[256,256,3])

    downsampling_layers = [downsample(64, 4, apply_instancenorm=False), #(128, 128, 64)
          downsample(128, 4), #(64, 64, 128)
          downsample(256, 4) # (32, 32, 256)
          ]

    upsampling_layers = [upsample(256, 4), # (64, 64, 256)
          upsample(128, 4) # (128, 128, 256)
        ]

    initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.02)
    last = layers.Conv2DTranspose(3, 4, strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  use_bias=False,
                                  activation = 'tanh')

    x = inputs

    for ds in downsampling_layers:
        x = ds(x)

    for us in upsampling_layers:
        x = us(x)

    return keras.Model(inputs=inputs, outputs=last(x))


def Discriminator():
    inputs = layers.Input(shape=[256, 256, 3])

    downsampling_layers = [downsample(64, 4, apply_instancenorm=False),  # (128, 128, 64)
                           downsample(128, 4),  # (64, 64, 128)
                           downsample(256, 4),  # (32, 32, 256)
                           downsample(512, 4),  # (16, 16, 512)
                           downsample(512, 4),  # (8, 8, 512)
                           downsample(512, 4),  # (4, 4, 512)
                           downsample(512, 2),  # (2, 2, 512)
                           downsample(512, 2)  # (1, 1, 512)
                           ]

    last = layers.Dense(1, activation='sigmoid')

    x = inputs

    for ds in downsampling_layers:
        x = ds(x)

    return keras.Model(inputs=inputs, outputs=last(x))


def generator_loss(generated):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.ones_like(generated), generated)

def discriminator_loss(real, generated):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.ones_like(real), real)

    generator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.zeros_like(generated), generated)

    return real_loss + generator_loss


def cycle_loss(image, tt_image, lambda_adj):
    mae = tf.reduce_mean(abs(image - tt_image))

    return lambda_adj * mae

def id_loss(image, id_image, lambda_adj):
    mae = tf.reduce_mean(abs(image - id_image))

    return lambda_adj * 0.5 * mae





# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
Dig_MNIST = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")


# train = pd.read_csv('Kannada-MNIST/train.csv')
# test = pd.read_csv('Kannada-MNIST/test.csv')
# Dig_MNIST = pd.read_csv("Kannada-MNIST/Dig-MNIST.csv")


X=train.iloc[:,1:].values
Y=train.iloc[:,0].values
Y[:10]


X = X.reshape(X.shape[0], 28, 28,1)
print(X.shape)


Y = keras.utils.to_categorical(Y, 10)
print(Y.shape)


x_test=test.drop('id', axis=1).iloc[:,:].values
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
x_test.shape


x_dig=Dig_MNIST.drop('label',axis=1).iloc[:,:].values
print(x_dig.shape)
x_dig = x_dig.reshape(x_dig.shape[0], 28, 28,1)
x_dig.shape


y_dig=Dig_MNIST.label
y_dig.shape


X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.10, random_state=42)


train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 10,
                                   width_shift_range = 0.40,
                                   height_shift_range = 0.25,
                                   shear_range = 0.1,
                                   zoom_range = 0.25,
                                   horizontal_flip = False)

valid_datagen = ImageDataGenerator(rescale=1./255)


initial_learningrate=2e-3
batch_size = 1024
epochs = 60


def lr_decay(epoch):#lrv
    return initial_learningrate * 0.99 ** epoch


from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('BWeight.md5', monitor='val_loss', save_best_only=True)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()


es = EarlyStopping(monitor='val_loss', verbose=2, patience=10)


model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=initial_learningrate),  metrics=['accuracy'])


from sklearn import metrics


history = model.fit_generator(
      train_datagen.flow(X_train,Y_train, batch_size=batch_size),
      steps_per_epoch=100,
      epochs=epochs,
      callbacks=[LearningRateScheduler(lr_decay),es,checkpoint],
      validation_data=valid_datagen.flow(X_valid,Y_valid),
      validation_steps=50,
      verbose=2)


model.load_weights('BWeight.md5')


preds_dig = model.predict_classes(x_dig / 255)
metrics.accuracy_score(preds_dig, y_dig)

submission = pd.DataFrame({'id': Id, 'label': preds_dig})
submission.to_csv(path_or_buf="submission.csv", index=False)


accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Test accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()


plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Test loss')
plt.title('Loss')
plt.legend()
plt.show()
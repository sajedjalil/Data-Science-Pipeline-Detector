# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import tensorflow as ft

from tensorflow import keras
from keras import layers

from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images / 255.0

model_mnist = keras.models.Sequential([
                                       layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
                                       layers.MaxPooling2D(2,2),
                                       layers.Conv2D(64, (3,3), activation="relu"),
                                       layers.MaxPooling2D(2,2),
                                       layers.Conv2D(64, (3,3), activation="relu"),
                                       layers.MaxPooling2D(2,2),
                                       layers.Flatten(),
                                       # layers.Dropout(0.1),
                                        layers.BatchNormalization(),
                                       layers.Dense(128, activation="elu", kernel_initializer="he_normal"),
                                        layers.BatchNormalization(),
                                       layers.Dense(10, activation="softmax")])

model_mnist.compile(loss="sparse_categorical_crossentropy", 
                    optimizer="adam", 
                    metrics=["accuracy"])

history = model_mnist.fit(train_images, train_labels, epochs=50)

test = pd.read_csv("../input/digit-recognizer/test.csv")

test = test/255.0

test = test.values.reshape(-1, 28, 28, 1)

predictions = model_mnist.predict_classes(test, verbose=0)

submissions = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
    "Label": predictions})
submissions.to_csv("submission.csv", index=False, header=True)
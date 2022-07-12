# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 20:33:58 2020

@author: Vithal Nistala
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/
from keras.layers import  Flatten
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D
image_size = 784 # 28*28
#num_classes = 10 # ten unique digits
df = pd.read_csv('../input/digit-recognizer/train.csv')
dt=pd.read_csv('../input/digit-recognizer/test.csv')
#print(dt.shape)
#
#print(dt.head(5))

x_train = (df.iloc[:,1:].values).astype('float32') # all pixel values
y_train = df.iloc[:,0].values.astype('int32') # only labels i.e targets digits
x_test =(dt.values).astype('float32')
#print(x_test.shape)
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
print(x_train.shape)
print(x_test.shape)

y_train= to_categorical(y_train)
num_classes = y_train.shape[1]
# Create the model: model
model = Sequential()
model.add(Conv2D(filters=5, kernel_size=5, padding='same', activation='relu', 
                        input_shape=(28, 28, 1)))
# Add a max pooling layer
model.add(MaxPooling2D(pool_size=4))
# Add a convolutional layer
model.add(Conv2D(filters=15,kernel_size=5, padding = 'same', activation ='relu' ))
# Add another max pooling layer
model.add(MaxPooling2D(pool_size=4))
# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Summarize the model
model.summary()



model.compile(optimizer='rmsprop', loss='categorical_crossentropy' , metrics=['accuracy'])

# Fit the model
model.fit(x_train,y_train,epochs=70, 
    shuffle=True, 
    verbose=2)
predictions = model.predict_classes(x_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("submissions.csv", index=False, header=True)




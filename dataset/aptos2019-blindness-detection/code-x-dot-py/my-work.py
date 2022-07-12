from random import shuffle 
import numpy as np
import pandas as pd 
import os
import cv2
import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

df = pd.read_csv('../input/train.csv')
y_train = np.array(df['diagnosis'])
#print(y_train)
def train_data_maker():
    x_train = []
    for i in tqdm.tqdm(sorted(os.listdir('../input/train_images'))):
        img_path = os.path.join('../input/train_images',i)
        image_array = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (50,50))
        x_train.append([np.array(image_array)])
    #np.save('x_train.npy',x_train)
    return np.array(x_train)
    


def neural_network(x_train,y_train,epochs_num=10):
    model = Sequential()

    model.add(Conv2D(64, (5, 5), input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(32, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))


    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(.5))
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(.5))
    model.add(Dense(5, activation=tf.nn.softmax))
    

    model.compile(optimizer='adam',  
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs_num)
    return model
#train_data_maker()
#x_train = np.load('x_train.npy')
x_train = train_data_maker()
x_train = x_train.reshape(3662, 50, 50, 1)
zipped = zip(x_train, y_train)
zipped = list(zipped)
shuffle(zipped)
x_train, y_train = zip(*zipped)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = x_train[-200:]
y_test = y_train[-200:]
mod = neural_network(x_train,y_train,epochs_num=20)
loss, acc = mod.evaluate(x_test, y_test)
print(acc)
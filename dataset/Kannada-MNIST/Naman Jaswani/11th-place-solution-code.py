

import csv
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras import layers,regularizers
import pandas as pd
from keras.optimizers import Adam,RMSprop


decay=1e-4
xtrain = []
ytrain = []
xtest  = []
#rough  = []
xval   = []
yval   = []


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
train =pd.read_csv(os.path.join(dirname,'train.csv'))
# print("train",np.shape(train))
test =pd.read_csv(os.path.join(dirname,'test.csv'))
# print("test",np.shape(test))
sample_submission =pd.read_csv(os.path.join(dirname,'sample_submission.csv')) 
Dig_MNIST=pd.read_csv(os.path.join(dirname,'Dig-MNIST.csv'))

xtrain=train.drop('label',axis=1)
ytrain=train.label
xtest = test.drop('id', axis = 1)
xval=Dig_MNIST.drop('label',axis=1)
yval=Dig_MNIST.label


xtrain=xtrain.values
xtest=xtest.values
xval=xval.values
ytrain=ytrain.values
yval=yval.values


xtrain=xtrain.reshape(-1,28,28,1).astype('float32')
xtest=xtest.reshape(-1,28,28,1).astype('float32')
xval=xval.reshape(-1,28,28,1).astype('float32')
# print(np.shape(xtest))

xtrain=(xtrain-np.mean(xtrain))/255
xtest=(xtest-np.mean(xtest))/255
xval=(xval-np.mean(xval))/255



model=Sequential()

model.add(layers.Conv2D(64, (3,3), padding='same', input_shape=(28, 28, 1)))
model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.Conv2D(64,  (3,3), padding='same'))
model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.Conv2D(64,  (3,3), padding='same'))
model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(layers.LeakyReLU(alpha=0.1))

model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, (3,3), padding='same'))
model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.Conv2D(128, (3,3), padding='same'))
model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.Conv2D(128, (3,3), padding='same'))
model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(layers.LeakyReLU(alpha=0.1))

model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.2)) 

model.add(layers.Conv2D(256, (3,3), padding='same'))
model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.Conv2D(256, (3,3), padding='same'))
model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(layers.LeakyReLU(alpha=0.1))

model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(256))
model.add(layers.LeakyReLU(alpha=0.1))

model.add(layers.BatchNormalization())
model.add(layers.Dense(256))
model.add(layers.LeakyReLU(alpha=0.1))

model.add(layers.BatchNormalization())
model.add(layers.Dense(10, activation='softmax'))


adam_opt = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999,  decay=0.0, amsgrad=False)
optimizer = RMSprop(learning_rate=0.002,rho=0.9)#,momentum=0.1,epsilon=1e-07,centered=True,name='RMSprop')
model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=12, 
    zoom_range=0.35, 
    width_shift_range=0.3, 
    height_shift_range=0.3,
#     rescale=1./255
)

datagen.fit(xtrain)

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau( 
    monitor='loss',    # Quantity to be monitored.
    factor=0.2,       # Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience=2,        # The number of epochs with no improvement after which learning rate will be reduced.
    verbose=1,         # 0: quiet - 1: update messages.
    mode="min",       # {auto, min, max}. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; 
                       # in the max mode it will be reduced when the quantity monitored has stopped increasing; 
                       # in auto mode, the direction is automatically inferred from the name of the monitored quantity.
    min_delta=0.0001,  # threshold for measuring the new optimum, to only focus on significant changes.
    cooldown=0,        # number of epochs to wait before resuming normal operation after learning rate (lr) has been reduced.
    min_lr=0.00001     # lower bound on the learning rate.
    )

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300, restore_best_weights=True)
model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=1024),
                              steps_per_epoch=len(xtrain)//1024,
                              epochs=50,
                              validation_data=(np.array(xval),np.array(yval)),
                              validation_steps=50,
                              callbacks=[learning_rate_reduction, es])



model.summary()
score=model.evaluate(np.array(xval),np.array(yval),batch_size=1024)

ytest = model.predict_classes(xtest)
id_col = np.arange(ytest.shape[0])
submission = pd.DataFrame({'id': id_col, 'label': ytest})
submission.to_csv('submission.csv', index = False)

print("accuracy",score[1])
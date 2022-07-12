
import os
print(os.listdir("../input"))

# This model was taken form Vijayabhaskar J job  - https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
# and with minor changes adjusted to this competition.

#1 step - loading libraries
from keras.models import Sequential
#Import from keras_preprocessing not from keras.preprocessing
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

# 2 step Loading data, attaching extention 'tif' to files
def append_ext(fn):
    return fn+".tif"

traindf=pd.read_csv("../input/train_labels.csv",dtype=str)
testdf=pd.read_csv("../input/sample_submission.csv",dtype=str)
testdf1=testdf["id"].copy()
traindf["id"]=traindf["id"].apply(append_ext)
testdf["id"]=testdf["id"].apply(append_ext)

# 3 step Preparing data - splitting to train and validation set, rescaling data
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

# 4 preprocessing data - all this datagen parameters you can copy from keras page https://keras.io/preprocessing/image/
train_generator=datagen.flow_from_dataframe(
                                            dataframe=traindf,
                                            directory="../input/train/",
                                            x_col="id",
                                            y_col="label",
                                            subset="training",
                                            batch_size=256,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="categorical",
                                            target_size=(96,96))

valid_generator=datagen.flow_from_dataframe(
                                            dataframe=traindf,
                                            directory="../input/train/",
                                            x_col="id",
                                            y_col="label",
                                            subset="validation",
                                            batch_size=256,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="categorical",
                                            target_size=(96,96))
test_datagen=ImageDataGenerator(rescale=1./255.)
# there are some tricks with test_datagen - y_col should be set to None, and shuffle to False - we need not to break order
test_generator=test_datagen.flow_from_dataframe(
                                                dataframe=testdf,
                                                directory="../input/test/",
                                                x_col="id",
                                                y_col=None,
                                                batch_size=256,
                                                seed=42,
                                                shuffle=False,
                                                class_mode=None,
                                                target_size=(96,96))

#5 step - building model
#model activatio
model = Sequential()
#first Convolutive layer - you choose parameters for Convolute filter (32, (3,3)). Input shape should be reversed and used as (96,96,3) 
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(96,96,3)))
model.add(Activation('relu'))
# some more layers added - how many should be - I don't know, you should balance between precision and overfitting
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))

#model.add(Conv2D(256, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# adding last layers - flattening and Dence
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

#6 step - model training 
model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])



STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
# there is a problem, choosing step size for test data. Number of test samples should be divisible exactly by
# step size, otherwise Kaggle would not accept this desision. That is why I skipped this parameter for
# model.predict_generator

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=20
)

model.evaluate_generator(generator=valid_generator
)
# 7 step - prediction - good idea to reset test_generator from previous jobs
test_generator.reset()
pred=model.predict_generator(test_generator,

                             verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

#8 step - preparing results in pandas form, converting to csv format for submission
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
for i, num in enumerate(filenames):
    filenames[i] = num.replace('.tif','')
#.split('.')[0]
results=pd.DataFrame({"id":filenames,
                      "label":predictions})
results["id"]=results["id"]   
results.to_csv("results.csv",index=False)
# Thank you
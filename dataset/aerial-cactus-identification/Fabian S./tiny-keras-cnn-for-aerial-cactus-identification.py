import os
import numpy as np
import pandas as pd
from keras import Sequential
from keras.preprocessing import image
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

train_data = pd.read_csv("../input/train.csv")
train_data['has_cactus'] = train_data['has_cactus'].astype('str')
train_dir = '../input/train/train/'
test_dir = '../input/test/'

#keras CNN model (the shape of the X_train has to be (17500,32,32,3))
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1)) #NN for classification task
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy']) #compile model

#image augmentation
img_gen = image.ImageDataGenerator(
    rescale=1/255,
    validation_split=0.10,
    horizontal_flip=True,
    vertical_flip=True)  

train_generator = img_gen.flow_from_dataframe(
    dataframe = train_data, 
    directory = train_dir, 
    x_col="id", 
    y_col="has_cactus",
    target_size=(32,32),
    subset="training",
    batch_size=32,
    shuffle=True,
    class_mode="binary"
)  

validation_generator = img_gen.flow_from_dataframe(
    dataframe = train_data,
    directory = train_dir,
    x_col="id",
    y_col="has_cactus",
    target_size=(32,32),
    subset="validation",
    batch_size=32,
    shuffle=True,
    class_mode="binary"
)

img_gen2 = image.ImageDataGenerator(
    rescale=1/255
)

test_generator = img_gen2.flow_from_directory(
    test_dir,
    target_size=(32,32),
    batch_size=1,
    shuffle=False,
    class_mode=None
)

filepath = 'ModelCheckpoint.h5'

callbacks = [
    ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1),
    EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    ]

history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,validation_data=validation_generator,
    validation_steps=len(validation_generator),
    verbose=2,
    callbacks=callbacks
)

#load best weights for final prediction
model.load_weights('ModelCheckpoint.h5')

y_pred = model.predict_generator(
    test_generator,
    steps=len(test_generator.filenames)
)

submission = pd.DataFrame()
submission['id'] = [test_file_name for test_file_name in sorted(os.listdir('../input/test/test/'))]
print(submission['id'])
submission['has_cactus'] = y_pred
submission.to_csv('submission.csv', header=True, index=False)
print(y_pred)
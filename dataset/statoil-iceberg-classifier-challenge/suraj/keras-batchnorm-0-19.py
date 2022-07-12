# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Dropout, MaxPooling2D, Flatten, GlobalMaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator

train = pd.read_json("../input/train.json")
test  = pd.read_json("../input/test.json")

X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)

X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
Test = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)

target = train['is_iceberg']
ID = test['id']

x_train, x_test, y_train, y_test = train_test_split(X,target,test_size=0.25,stratify=target,random_state=10)

datagen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.,
                         height_shift_range = 0.,
                         channel_shift_range=0,
                         zoom_range = 0.2,
                         rotation_range = 10)

model = Sequential()

model.add(BatchNormalization(input_shape=(75,75,3)))

model.add(Conv2D(32, kernel_size=(3, 3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128,kernel_size=(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(GlobalMaxPooling2D())

model.add(Dense(64)) #512
model.add(Activation('relu'))


model.add(Dense(1))
model.add(Activation('sigmoid'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

check = ModelCheckpoint("weights.{epoch:02d}-{val_acc:.5f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='max')

model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),steps_per_epoch=len(x_train)/32,epochs=1,callbacks=[check,early],validation_data=(x_test,y_test))

pred = model.predict_proba(Test)

submission = pd.DataFrame()
submission['id'] = ID
submission['is_iceberg'] = pred
submission.to_csv('submissions.csv', index=False)
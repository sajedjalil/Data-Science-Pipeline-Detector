import numpy as np 
import pandas as pd 

from subprocess import check_output

import os, cv2, random
import numpy as np
import pandas as pd



from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils




TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'


ROWS = 30
COLS = 30
CHANNELS = 1


train_image = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] 


test_image =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i%250 == 0: print('Processed {} of {}'.format(i, count))
    
    return data

train = prep_data(train_image)
test = prep_data(test_image)

print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))

labels = []
for i in train_image:
    if 'dog' in i:
        labels.append(1)
    else:
        labels.append(0)
        
train = train.reshape(-1, 32,32,1)
test = test.reshape(-1, 32,32,1)
X_train = train.astype('float32')
X_test = test.astype('float32')
X_train /= 255
X_test /= 255
Y_train=labels

X_valid = X_train[:5000,:,:,:]
Y_valid =   Y_train[:5000]
X_train = X_train[5001:25000,:,:,:]
Y_train  = Y_train[5001:25000]

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)



model = Sequential()
model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=(ROWS, COLS, CHANNELS), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))
                                 
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=128, nb_epoch=8,
          show_accuracy=True, verbose=1,
          validation_data=(X_valid, Y_valid))
          
submission = model.predict_proba(X_test, verbose=1)
test_id = range(1,12575)
predictions_df = pd.DataFrame({'id': test_id, 'label': submission[:,0]})

predictions_df.to_csv("submission.csv", index=False)

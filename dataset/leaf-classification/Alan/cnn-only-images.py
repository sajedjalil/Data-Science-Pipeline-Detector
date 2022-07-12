import pandas as pd
# from scipy.misc import imread
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.utils import np_utils

def load_images():
    dr = "../input/images/"
    ims = [Image.open(dr + f) for f in os.listdir(dr) if f.endswith('jpg')]
    max_height = max(i.height for i in ims)
    max_width = max(i.width for i in ims)
    newims = []
    img_rows, img_cols = 0, 0
    for im in ims:
        new_im = im.crop((0, 0, max_width, max_height))
        new_im.thumbnail((100,100))
        img_rows, img_cols = max(img_rows, new_im.height), max(img_cols, new_im.width)
        newims.append(np.asarray(new_im))
    print(img_rows,img_cols)
    return np.stack(newims),img_rows, img_cols

train = pd.read_csv('../input/train.csv')
y_raw = train.pop('species')
y = LabelEncoder().fit(y_raw).transform(y_raw)
Y_train = np_utils.to_categorical(y)

train_ids = train.pop('id')
test_ids = pd.read_csv('../input/test.csv').pop('id')

ims,img_rows, img_cols = load_images()

X_train = ims[train_ids - 1]
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = ims[test_ids - 1]
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

batch_size = 128
nb_classes = 10
nb_epoch = 12

input_shape = (1,img_rows,img_cols)
kernel_size = (3, 3)
pool_size = (2, 2)
nb_classes = len(np.unique(y))
nb_filters = 32
print(nb_classes)

input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_train /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')



model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=10)
print(history.history['loss'])
yPred = model.predict_proba(X_test)
yPred = pd.DataFrame(yPred,index=index,columns=sort(parent_data.species.unique()))
fp = open('submission_cnn_kernel.csv','w')
fp.write(yPred.to_csv())

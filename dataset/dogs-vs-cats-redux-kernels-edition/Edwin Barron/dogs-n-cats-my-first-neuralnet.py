import os, cv2
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import csv

from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.utils import np_utils

# Most of this first bit was taken from https://www.kaggle.com/jeffd23/dogs-vs-cats-redux-kernels-edition/catdognet-keras-convnet-starter/notebook

train_data = ['../input/train/' + i for i in os.listdir('../input/train/')]
test_data = ['../input/test/' + i for i in os.listdir('../input/test/')]

labels = []
for i in train_data:
    if 'dog' in i:
        labels.append(1)
    else:
        labels.append(0)

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img,(32,32),interpolation=cv2.INTER_CUBIC)#Original was 64/64, but in class 32/32 was recommended.

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, 3,32,32), dtype=np.uint8)
    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i%250 == 0: print('Processed {} of {}'.format(i, count))
    return data

train_data =prep_data(train_data)
test_data =prep_data(test_data)

#This is where I started using stuff from https://github.com/wxs/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb
Model = Sequential()
Model.add(Convolution2D(16,3,3,border_mode= 'same', input_shape = (3,32,32), activation = 'relu'))
Model.add(MaxPooling2D(pool_size = (2,2), dim_ordering="th"))
Model.add(Convolution2D(32,3,3,border_mode = 'same', activation='relu'))#typically double the number of filters each new layer (32 this time).
Model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
Model.add(Flatten())
Model.add(Dense(100,activation='relu'))
Model.add(Dropout(0.5))
Model.add(Dense(100,activation='relu'))
Model.add(Dropout(0.5))
Model.add(Dense(1))
Model.add(Activation('sigmoid'))
Model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])



Model.fit(train_data, labels,
          batch_size=16, nb_epoch=2,
          verbose=1)

predicted_classes = Model.predict_classes(test_data)
for i in predicted_classes:
    print(i)
"""with open('../predict.csv', 'w', newline='') as csvfile:
    cdwrite = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in predicted_classes:
        cdwrite.writerow(i)"""
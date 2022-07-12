import os
import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from sklearn.metrics import mean_squared_error
#RMSE = mean_squared_error(y, y_pred)**0.5

train_data = pd.read_csv('../input/Train/train.csv')
train_imgs = sorted(glob.glob('../input/Train/*.jpg'), key=lambda name: int(os.path.basename(name)[:-4]))
train_dot_imgs = sorted(glob.glob('../input/TrainDotted/*.jpg'), key=lambda name: int(os.path.basename(name)[:-4]))
os.system('ls ../input/Test')
test_imgs = sorted(glob.glob('../input/Test/*.jpg'), key=lambda name: int(os.path.basename(name)[:-4]))

print('Number of Train Images: {:d}'.format(len(train_imgs)))
print('Number of Dotted-Train Images: {:d}'.format(len(train_dot_imgs)))
print('Number of Test Images: {:d}'.format(len(test_imgs)))

print(train_data.head(6))

# 5. Preprocess input data
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
# X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255

# # 6. Preprocess class labels
# Y_train = np_utils.to_categorical(y_train, 10)
# Y_test = np_utils.to_categorical(y_test, 10)
 
# # 7. Define model architecture
# model = Sequential()
 
# model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
# model.add(Convolution2D(32, 3, 3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
 
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))
 
# # 8. Compile model
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
 
# # 9. Fit model on training data
# model.fit(X_train, Y_train, 
#           batch_size=32, nb_epoch=10, verbose=1)
 
# # 10. Evaluate model on test data
# score = model.evaluate(X_test, Y_test, verbose=0)
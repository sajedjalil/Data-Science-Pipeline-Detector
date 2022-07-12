import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib.pyplot as plt
#import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from matplotlib import pyplot
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, BatchNormalization, Flatten, MaxPool2D
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
y_train = train_data['label']
X = train_data.drop(['label'], axis=1)
del train_data

Id = test_data['id']
test_data = test_data.drop(['id'], axis=1)

#label_val = y_train.value_counts()

#X_temp = X.values.reshape(X.shape[0], 28, 28)

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
num_classes = len(classes)
#samples_per_class = 6

#transformando imagem de entrada que contem 28x28
X = X.values.reshape(X.shape[0], 28, 28, 1)
test_data = test_data.values.reshape(test_data.shape[0], 28, 28, 1)


datagen = ImageDataGenerator(zoom_range=0.12, shear_range=0.12 , width_shift_range=0.5, height_shift_range=0.1)
#datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, zoom_range=0.03, shear_range=0.03)
datagen.fit(X)

y_train = to_categorical(y_train, num_classes=10)


model = Sequential()
model.add(Conv2D(28, kernel_size=3, strides=1, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(28, kernel_size=3, strides=1, activation='relu', padding='same'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(28, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))
model.add(Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=3, strides=1,padding='same', activation='relu'))
model.add(Dropout(0.4))


model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))


model.summary()


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


X_train, X_val1, y_train, y_val1 = train_test_split(X, y_train, test_size=0.05, random_state=42)

size_batch = 64
epoch = 60

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=size_batch), epochs=epoch, validation_data=(X_val1, y_val1), verbose=2, steps_per_epoch=X_train.shape[0] // size_batch)
#history = model.fit(datagen.flow(X_train, y_train, batch_size=size_batch), epochs=epoch, validation_data=(X_val1, y_val1), verbose=2, steps_per_epoch=X_train.shape[0] // size_batch)


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


FINAL_PREDS = model.predict_classes(test_data)

submission = pd.DataFrame({'id': Id, 'label': FINAL_PREDS})
submission.to_csv(path_or_buf="submission.csv", index=False)


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import keras.preprocessing.image as image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, LeakyReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def load_dataset(image_dir, label_file):
    labels = pd.read_csv(label_file)
    x = []
    for img_name in labels['id']:
        image_path = os.path.join(image_dir, img_name)
        pil_image = image.load_img(image_path, target_size=(32, 32, 3))
        np_image = image.img_to_array(pil_image)
        x.append(np_image)
    # normalizing pixel values
    x = np.array(x).astype('float32') / 255
    y = labels['has_cactus']
    return x, y
    
def simple_cnn(input_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, padding="same", input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(Conv2D(filters=32, kernel_size=3, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(Conv2D(filters=64, kernel_size=3, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(Conv2D(filters=128, kernel_size=3, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units=500, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(units=1, activation="sigmoid"))

    model.summary()
    return model
    
# TRAINING 
tr_dir = "../input/train/train/"
tr_labels_file = "../input/train.csv"
x_train, y_train = load_dataset(tr_dir, tr_labels_file)
# splitting training and validation
(x_val, x_train) = x_train[:5000], x_train[5000:]
(y_val, y_train) = y_train[:5000], y_train[5000:]
model = simple_cnn()

# Callbacks
checkpointer = ModelCheckpoint("./model.best.hdf5", monitor="val_loss", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor="val_loss", min_delta=0.0, patience=20, verbose=1)
lratereducer = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, verbose=1)

# train the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])#, metrics.precision, metrics.recall])

history = model.fit(x_train, y_train,  validation_data=(x_val, y_val), batch_size=64, epochs=100, verbose=2,
          callbacks=[checkpointer, earlystopper, lratereducer], shuffle=True)

# plotting losses and accuracies
best_val_loss_epoch = np.argmin(history.history["val_loss"])
best_val_acc_epoch = np.argmax(history.history["val_acc"])

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(["train", "val"], loc='lower left')
plt.axvline(best_val_loss_epoch)
plt.show()

plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model acc")
plt.ylabel("acc")
plt.xlabel("epochs")
plt.legend(["train", "val"], loc='lower left')
plt.axvline(best_val_acc_epoch)
plt.show()

plt.plot(history.history["lr"])
plt.title("Learning rate")
plt.ylabel("lr")
plt.xlabel("epochs")
plt.show()

# PREDICT
model = simple_cnn()
ts_dir = "../input/test/test/"
ts_labels_file = "../input/sample_submission.csv"
x_test, y_dummy = load_dataset(ts_dir, ts_labels_file)
model.load_weights("model.best.hdf5")
y_predictions = model.predict(x_test)
df = pd.DataFrame({'id' : pd.read_csv(ts_labels_file)['id'],
                   'has_cactus' : y_predictions.squeeze()})
df.to_csv("submission.csv", index=False, columns=['id', 'has_cactus'])

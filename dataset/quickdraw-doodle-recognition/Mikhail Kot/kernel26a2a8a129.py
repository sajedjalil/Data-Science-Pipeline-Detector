import zipfile
import numpy as np
import pandas as pd
import os
import tqdm
import matplotlib

import matplotlib.pyplot as plt
import csv
import ast
from PIL import Image, ImageDraw
import cv2
import tensorflow as tf
import keras
import io
import os

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.models import save_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, GlobalAveragePooling2D, \
    BatchNormalization, LeakyReLU
import keras.backend.tensorflow_backend as K

K.set_session

train_path = "/kaggle/input/quickdraw-doodle-recognition/train_simplified"
train_files = []

for dirname, _, filenames in os.walk(train_path):
    for filename in filenames:
        joined = os.path.join(dirname, filename)
        
        print(joined)
        train_files.append(joined)

class_labels = [e.split('/')[-1].split('.')[0] for e in train_files]

NUM_CLASSES = len(class_labels)
class_to_idx = {c: idx for idx, c in enumerate(class_labels)}

print(NUM_CLASSES)
print(class_labels[:10])

def get_eternal_csv_generator(fn, debug=False):
    while True:
        with open(fn) as f:
            f.readline()  # skip header
            for line in csv.reader(f, delimiter=',', quotechar='"'):
                yield line[1], line[5]
                
def raw_batch_generator(batch_size, debug=False):
    generators = np.array([get_eternal_csv_generator(fn, debug) for fn in train_files])
    while True:
        random_indices = np.random.randint(0, len(generators), size=batch_size)
        yield [next(gen) for gen in generators[random_indices]]
        
IMG_SIZE = 128

def draw_it(strokes):
    img = 255 * np.ones((256, 256), np.uint8)
    for stroke in ast.literal_eval(strokes):
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), 0, 3)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))

def images_and_labels_generator(batch_size):
    for batch in raw_batch_generator(batch_size):
        batch_images = []
        batch_labels = []
        for e in batch:
            batch_images.append(draw_it(e[0]))
            batch_labels.append(e[1])
        batch_images = np.stack(batch_images, axis=0)
        yield batch_images, batch_labels

def train_iterator(batch_size):
    for batch in images_and_labels_generator(batch_size):
        images = batch[0].astype('float32')
        
        images = images / 256 - 0.5
        
        images = np.expand_dims(images, -1)
        labels = keras.utils.to_categorical(list(map(class_to_idx.get, batch[1])), NUM_CLASSES)
        yield images, labels

def make_model():
    model = Sequential()
    
    model.add(Conv2D(64, (7, 7), padding='same', activation='elu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='elu'))
    
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='elu'))
    
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding='same', activation='elu'))

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding='same', activation='elu'))
    
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding='same', activation='elu'))
    
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding='same', activation='elu'))

    model.add(GlobalAveragePooling2D()) 

    model.add(Dropout(0.2))
    model.add(Dense(NUM_CLASSES, activation="softmax"))
    model.add(LeakyReLU())
    
    return model

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def test_csv_iterator(batch_size):
    with open("/kaggle/input/quickdraw-doodle-recognition/test_simplified.csv", "r") as f:
        batch_keys = []
        batch_images = []
        f.readline()  # skip header
        for line in csv.reader(f, delimiter=',', quotechar='"'):
            batch_keys.append(line[0])
            batch_images.append(draw_it(line[2]))
            if len(batch_images) == batch_size:
                batch_images = np.stack(batch_images, axis=0)
                batch_images = np.expand_dims(batch_images, -1)
                batch_images = batch_images.astype('float32')
                
                batch_images = batch_images / 256 - 0.5
                
                yield batch_keys, batch_images
                batch_keys = []
                batch_images = []
        if batch_images:  # last batch
            batch_images = np.stack(batch_images, axis=0)
            batch_images = np.expand_dims(batch_images, -1)
            batch_images = batch_images.astype('float32')
            
            batch_images = batch_images / 256 - 0.5
            
            yield batch_keys, batch_images
            

model = make_model()

BATCH_SIZE = 200
STEPS_PER_EPOCH = 150
EPOCHS = 50

model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.adam(clipnorm=5.),
    metrics=[categorical_accuracy, top_3_accuracy] 
)

CHECKPOINT_TEMPLATE = "/model_{}"

model.fit_generator(
    train_iterator(BATCH_SIZE), 
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    verbose=1,
    initial_epoch=0
)

save_model(model, "model")
print("Model saved in {}".format("model"))

with open("submission.csv", "w", buffering=1*1024*1024) as f:
    f.write("key_id,word\n")
    for batch_keys, batch_images in tqdm.tqdm(test_csv_iterator(BATCH_SIZE), total=np.ceil(112200./BATCH_SIZE)):
        probas = model.predict_proba(batch_images, BATCH_SIZE)
        top_3_classes = np.argsort(probas, axis=1)[:, [-1, -2, -3]]
        labels = map(lambda x: " ".join("_".join(class_labels[idx].split()) for idx in x), top_3_classes)
        for key, labels in zip(batch_keys, labels):
            f.write(key + "," + labels + "\n")
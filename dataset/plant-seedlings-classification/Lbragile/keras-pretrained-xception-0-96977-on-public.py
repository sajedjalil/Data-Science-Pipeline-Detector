import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, cv2
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.applications import *

x_train = []
x_test = []
y_train = []

df_test = pd.read_csv('sample_submission.csv')


label_map = {   "Black-grass"               :0,
                "Charlock"                  :1,
                "Cleavers"                  :2,
                "Common Chickweed"          :3,
                "Common wheat"              :4,
                "Fat Hen"                   :5,
                "Loose Silky-bent"          :6,
                "Maize"                     :7,
                "Scentless Mayweed"         :8,
                "Shepherds Purse"           :9,
                "Small-flowered Cranesbill" :10,
                "Sugar beet"                :11}

dim = 256

# Preparing training data
dirs = os.listdir("train/")
for k in tqdm(range(len(dirs))):    # Directory
    files = os.listdir("train/{}".format(dirs[k]))
    for f in range(len(files)):     # Files
        img = cv2.imread('train/{}/{}'.format(dirs[k], files[f]))
        targets = np.zeros(12)
        targets[label_map[dirs[k]]] = 1 
        x_train.append(cv2.resize(img, (dim, dim)))
        y_train.append(targets)
    
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float32)

print(x_train.shape)
print(y_train.shape)

#x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.01, random_state=42)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.16, random_state=42) # Want a balanced split for all the classes
for train_index, test_index in sss.split(x_train, y_train):
    print("Using {} for training and {} for validation".format(len(train_index), len(test_index)))
    x_train, x_valid = x_train[train_index], x_train[test_index]
    y_train, y_valid = y_train[train_index], y_train[test_index]

datagen = ImageDataGenerator( horizontal_flip=True, 
                              vertical_flip=True)
                                      
weights = os.path.join('', 'weights.h5')

epochs = 100
learning_rate = 0.0001
batch_size = 32

callbacks = [ EarlyStopping(monitor='val_loss', patience=5, verbose=0), 
              ModelCheckpoint(weights, monitor='val_loss', save_best_only=True, verbose=0),
              ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)]

base_model = Xception(input_shape=(dim, dim, 3), include_top=False, weights='imagenet', pooling='avg') # Average pooling reduces output dimensions
x = base_model.output
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(12, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layers not in classifier due to loading imagenet weights
# for layer in base_model.layers:
#     layer.trainable = False

# print(model.summary())

# Load any existing weights
# if os.path.isfile(weights):
#     model.load_weights(weights)
    
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

# ------ TRAINING ------
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train)/batch_size, 
                    validation_data=datagen.flow(x_valid, y_valid, batch_size=batch_size), 
                    validation_steps=len(x_valid)/batch_size,
                    callbacks=callbacks,
                    epochs=epochs, 
                    verbose=1)
                    
# ------ TESTING ------
for f, species in tqdm(df_test.values, miniters=100):
    img = cv2.imread('test/{}'.format(f))
    x_test.append(cv2.resize(img, (dim, dim)))

x_test = np.array(x_test, np.float32)
print(x_test.shape)

if os.path.isfile(weights):
    model.load_weights(weights)

p_test = model.predict(x_test, verbose=1)

preds = []
for i in range(len(p_test)):
    pos = np.argmax(p_test[i])
    preds.append(list(label_map.keys())[list(label_map.values()).index(pos)])
    
df_test['species'] = preds
df_test.to_csv('submission.csv', index=False)
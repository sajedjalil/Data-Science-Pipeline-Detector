import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import cv2
from tqdm import tqdm

from sklearn.metrics import fbeta_score

# Params
input_size = 64
input_channels = 3

epochs = 15
batch_size = 128
learning_rate = 0.001
lr_decay = 1e-4

valid_data_size = 5000  # Samples to withhold for validation

model = Sequential()
model.add(BatchNormalization(input_shape=(input_size, input_size, input_channels)))
model.add(Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))

df_train_data = pd.read_csv('../input/train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train_data['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

x_valid = []
y_valid = []

df_valid = df_train_data[(len(df_train_data) - valid_data_size):]

for f, tags in tqdm(df_valid.values, miniters=100):
    img = cv2.resize(cv2.imread('../input/train-jpg/{}.jpg'.format(f)), (input_size, input_size))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_valid.append(img)
    y_valid.append(targets)

y_valid = np.array(y_valid, np.uint8)
x_valid = np.array(x_valid, np.float32)

x_train = []
y_train = []

df_train = df_train_data[:(len(df_train_data) - valid_data_size)]

for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.resize(cv2.imread('../input/train-jpg/{}.jpg'.format(f)), (input_size, input_size))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(img)
    y_train.append(targets)
    img = cv2.flip(img, 0)  # flip vertically
    x_train.append(img)
    y_train.append(targets)
    img = cv2.flip(img, 1)  # flip horizontally
    x_train.append(img)
    y_train.append(targets)
    img = cv2.flip(img, 0)  # flip vertically
    x_train.append(img)
    y_train.append(targets)

y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float32)

df_test_data = pd.read_csv('../input/sample_submission_v2.csv')

x_test = []

for f, tags in tqdm(df_test_data.values, miniters=1000):
    img = cv2.resize(cv2.imread('../input/test-jpg/{}.jpg'.format(f)), (input_size, input_size))
    x_test.append(img)

x_test = np.array(x_test, np.float32)

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=5,
                           verbose=0),
             TensorBoard(log_dir='logs'),
             ModelCheckpoint('weights.h5',
                             save_best_only=True)]

opt = Adam(lr=learning_rate, decay=lr_decay)

model.compile(loss='binary_crossentropy',
              # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          callbacks=callbacks,
          validation_data=(x_valid, y_valid))

p_valid = model.predict(x_valid, batch_size=batch_size)
print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))

y_test = []

p_test = model.predict(x_test, batch_size=batch_size, verbose=2)
y_test.append(p_test)

result = np.array(y_test[0])
result = pd.DataFrame(result, columns=labels)

preds = []

for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    a = a.apply(lambda x: x > 0.2, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

df_test_data['tags'] = preds
df_test_data.to_csv('submission.csv', index=False)

# 0.918

from __future__ import division
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Merge, merge
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from multi_gpu import make_parallel #Available here https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import os
from sklearn.utils import shuffle
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import fbeta_score
from keras.optimizers import Adam, SGD

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def fbeta_loss(y_true, y_pred):
    beta_squared = 4

    tp = K.sum(y_true * y_pred) + K.epsilon()
    fp = K.sum(y_pred) - tp
    fn = K.sum(y_true) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    result = 1 - (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

    return result

def fbeta_score_K(y_true, y_pred):
    beta_squared = 4

    tp = K.sum(y_true * y_pred) + K.epsilon()
    fp = K.sum(y_pred) - tp
    fn = K.sum(y_true) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    result = (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

    return result

def rotate(img):
    rows = img.shape[0]
    cols = img.shape[1]
    angle = np.random.choice((10, 20, 30))#, 40, 50, 60, 70, 80, 90))
    rotation_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img = cv2.warpAffine(img, rotation_M, (cols, rows))
    return img

def rotate_bound(image, size):
    #credits http://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    angle = np.random.randint(10,180)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    output = cv2.resize(cv2.warpAffine(image, M, (nW, nH)), (size, size))
    return output

def perspective(img):
    rows = img.shape[0]
    cols = img.shape[1]

    shrink_ratio1 = np.random.randint(low=85, high=110, dtype=int) / 100
    shrink_ratio2 = np.random.randint(low=85, high=110, dtype=int) / 100

    zero_point = rows - np.round(rows * shrink_ratio1, 0)
    max_point_row = np.round(rows * shrink_ratio1, 0)
    max_point_col = np.round(cols * shrink_ratio2, 0)

    src = np.float32([[zero_point, zero_point], [max_point_row-1, zero_point], [zero_point, max_point_col+1], [max_point_row-1, max_point_col+1]])
    dst = np.float32([[0, 0], [rows, 0], [0, cols], [rows, cols]])

    perspective_M = cv2.getPerspectiveTransform(src, dst)

    img = cv2.warpPerspective(img, perspective_M, (cols,rows))#, borderValue=mean_pix)
    return img

def shift(img):
    rows = img.shape[0]
    cols = img.shape[1]

    shift_ratio1 = (random.random() * 2 - 1) * np.random.randint(low=3, high=15, dtype=int)
    shift_ratio2 = (random.random() * 2 - 1) * np.random.randint(low=3, high=15, dtype=int)

    shift_M = np.float32([[1,0,shift_ratio1], [0,1,shift_ratio2]])
    img = cv2.warpAffine(img, shift_M, (cols, rows))#, borderValue=mean_pix)
    return img

def batch_generator_train(zip_list, img_size, batch_size, is_train=True, shuffle=True):
    number_of_batches = np.ceil(len(zip_list) / batch_size)
    if shuffle == True:
        random.shuffle(zip_list)
    counter = 0
    while True:
        if shuffle == True:
            random.shuffle(zip_list)

        batch_files = zip_list[batch_size*counter:batch_size*(counter+1)]
        image_list = []
        mask_list = []

        for file, mask in batch_files:

            image = cv2.imread(file) #cv2.resize(cv2.imread(file), (img_size,img_size)) / 255.
            image = image[:, :, [2, 1, 0]] - mean_pix

            rnd_flip = np.random.randint(2, dtype=int)
            rnd_rotate = np.random.randint(2, dtype=int)
            rnd_zoom = np.random.randint(2, dtype=int)
            rnd_shift = np.random.randint(2, dtype=int)

            if (rnd_flip == 1) & (is_train == True):
                rnd_flip = np.random.randint(3, dtype=int) - 1
                image = cv2.flip(image, rnd_flip)

            if (rnd_rotate == 1) & (is_train == True):
                image = rotate_bound(image, img_size)

            if (rnd_zoom == 1) & (is_train == True):
                image = perspective(image)

            if (rnd_shift == 1) & (is_train == True):
                image = shift(image)

            image_list.append(image)
            mask_list.append(mask)

        counter += 1
        image_list = np.array(image_list)
        mask_list = np.array(mask_list)

        yield (image_list, mask_list)

        if counter == number_of_batches:
            if shuffle == True:
                random.shuffle(zip_list)
            counter = 0

def batch_generator_test(zip_list, img_size, batch_size, shuffle=True):
    number_of_batches = np.ceil(len(zip_list)/batch_size)
    print(len(zip_list), number_of_batches)
    counter = 0
    if shuffle:
        random.shuffle(zip_list)
    while True:
        batch_files = zip_list[batch_size*counter:batch_size*(counter+1)]
        image_list = []
        mask_list = []

        for file, mask in batch_files:

            image = cv2.resize(cv2.imread(file), (img_size, img_size))
            image = image[:, :, [2, 1, 0]] - mean_pix
            image_list.append(image)
            mask_list.append(mask)

        counter += 1
        image_list = np.array(image_list)
        mask_list = np.array(mask_list)

        yield (image_list, mask_list)

        if counter == number_of_batches:
            random.shuffle(zip_list)
            counter = 0

def predict_generator(files, img_size, batch_size):
    number_of_batches = np.ceil(len(files) / batch_size)
    print(len(files), number_of_batches)
    counter = 0
    int_counter = 0

    while True:
            beg = batch_size * counter
            end = batch_size * (counter + 1)
            batch_files = files[beg:end]
            image_list = []

            for file in batch_files:
                int_counter += 1
                image = cv2.resize(cv2.imread(file), (img_size, img_size))
                image = image[:, :, [2, 1, 0]] - mean_pix

                rnd_flip = np.random.randint(2, dtype=int)
                rnd_rotate = np.random.randint(2, dtype=int)
                rnd_zoom = np.random.randint(2, dtype=int)
                rnd_shift = np.random.randint(2, dtype=int)

                if rnd_flip == 1:
                    rnd_flip = np.random.randint(3, dtype=int) - 1
                    image = cv2.flip(image, rnd_flip)

                if rnd_rotate == 1:
                    image = rotate_bound(image, img_size)

                if rnd_zoom == 1:
                    image = perspective(image)

                if rnd_shift == 1:
                    image = shift(image)

                image_list.append(image)

            counter += 1

            image_list = np.array(image_list)

            yield (image_list)


def f2_score(y_true, y_pred):
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    score = fbeta_score(y_true, y_pred, beta=2, average='samples')
    return score

GLOBAL_PATH = 'D:/G/Amazon/'
TRAIN_FOLDER = 'D:/g/amazon/train-jpg-res/' #All train files resized to 224*224
TEST_FOLDER = 'D:/G/Amazon/test-jpg/' #All test files in one folder
F_CLASSES = GLOBAL_PATH + 'train_v2.csv'

df_train = pd.read_csv(F_CLASSES)
df_test = pd.read_csv(GLOBAL_PATH + 'sample_submission_v4.csv')

labels = ['blow_down',
          'bare_ground',
          'conventional_mine',
          'blooming',
          'cultivation',
          'artisinal_mine',
          'haze',
          'primary',
          'slash_burn',
          'habitation',
          'clear',
          'road',
          'selective_logging',
          'partly_cloudy',
          'agriculture',
          'water',
          'cloudy']
label_map = {'agriculture': 14,
             'artisinal_mine': 5,
             'bare_ground': 1,
             'blooming': 3,
             'blow_down': 0,
             'clear': 10,
             'cloudy': 16,
             'conventional_mine': 2,
             'cultivation': 4,
             'habitation': 9,
             'haze': 6,
             'partly_cloudy': 13,
             'primary': 7,
             'road': 11,
             'selective_logging': 12,
             'slash_burn': 8,
             'water': 15}

flatten = lambda l: [item for sublist in l for item in sublist]

x_train = []
x_test = []
y_train = []


for f, tags in tqdm(df_train.values, miniters=1000):
    img = TRAIN_FOLDER + '{}.jpg'.format(f)
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(img)
    y_train.append(targets)


x_train, x_holdout, y_train, y_holdout = x_train[3000:-1], x_train[:3000], y_train[3000:-1], y_train[:3000]

x_train, y_train = shuffle(x_train, y_train, random_state = 24)

part = 0.85
split = int(round(part*len(y_train)))
x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]
print('x tr: ', len(x_train))

#define callbacks
callbacks = [ModelCheckpoint('amazon_2007.hdf5', monitor='val_loss', save_best_only=True, verbose=2, save_weights_only=False),
             ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.0000001),
             EarlyStopping(monitor='val_loss', patience=5, verbose=0)]

BATCH = 128
IMG_SIZE = 224
mean_pix = np.array([102.9801, 115.9465, 122.7717]) #It is BGR

from keras.applications import ResNet50

#Compile model and set non-top layets non-trainable (warm-up)
base_model = ResNet50(include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3), pooling='avg', weights='imagenet')
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Dense(2048, activation='relu')(x)
x = Dropout(0.25)(x)
output = Dense(17, activation='sigmoid')(x)

optimizer = Adam(0.001, decay=0.0003)
model = Model(inputs=base_model.inputs, outputs=output)
model = make_parallel(model, 2)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', fbeta_score_K])

model.fit_generator(generator=batch_generator_train(list(zip(x_train, y_train)), IMG_SIZE, BATCH),
                          steps_per_epoch=np.ceil(len(x_train)/BATCH),
                          epochs=1,
                          verbose=1,
                          validation_data=batch_generator_train(list(zip(x_valid, y_valid)), IMG_SIZE, 16),
                          validation_steps=np.ceil(len(x_valid)/16),
                          callbacks=callbacks,
                          initial_epoch=0)


#Compile model and set all layers trainable
optimizer = Adam(0.0001, decay=0.00000001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', fbeta_score_K])
model.load_weights('amazon_2007.hdf5', by_name=True)
for layer in base_model.layers:
    layer.trainable = True

BATCH = 32
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', fbeta_score_K])
model.fit_generator(generator=batch_generator_train(list(zip(x_train, y_train)), IMG_SIZE, BATCH),
                          steps_per_epoch=np.ceil(len(x_train)/BATCH),
                          epochs=50,
                          verbose=1,
                          validation_data=batch_generator_train(list(zip(x_valid, y_valid)), IMG_SIZE, 16),
                          validation_steps=np.ceil(len(x_valid)/16),
                          callbacks=callbacks,
                          initial_epoch=0)

model.load_weights('amazon_2007.hdf5')


x_val = []
y_val = []
x_hld = []
y_hld = []
x_test = []
y_test = []


#====================== validation set est =================================
for f, tags in tqdm(list(zip(x_valid, y_valid)), miniters=1000):
    y_val.append(tags)

p_valid = model.predict_generator(batch_generator_test(list(zip(x_valid, y_valid)), IMG_SIZE, 8, shuffle=False), steps=np.ceil(len(x_valid)/8))

print('val_set: ', fbeta_score(np.array(y_val), np.array(p_valid) > 0.2, beta=2, average='samples'))
#===========================================================================

def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
    #credits https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475
  def mf(x):
    p2 = np.zeros_like(p)
    for i in range(17):
      p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    score = fbeta_score(y, p2, beta=2, average='samples')
    return score

  x = [0.2]*17
  for i in range(17):
    best_i2 = 0
    best_score = 0
    for i2 in range(resolution):
      i2 /= resolution
      x[i] = i2
      score = mf(x)
      if score > best_score:
        best_i2 = i2
        best_score = score
    x[i] = best_i2
    if verbose:
      print(i, best_i2, best_score)

  return x

X = optimise_f2_thresholds(np.array(y_val), np.array(p_valid))

#====================== holdout set est =================================
for f, tags in tqdm(list(zip(x_holdout, y_holdout)), miniters=1000):
    img = cv2.resize(cv2.imread(f), (IMG_SIZE, IMG_SIZE))
    x_hld.append(img)
    y_hld.append(tags)

if len(x_holdout) % 2 > 0:
    x_hld.append(x_hld[0])
    y_hld.append(y_hld[0])

x_hld = np.array(x_hld, np.float16)

p_valid = model.predict(x_hld, batch_size=28, verbose=2)
print('holdout set: ', f2_score(np.array(y_hld), np.array(p_valid) > 0.2))
print('holdout set w/ thresh: ', f2_score(np.array(y_hld), np.array(p_valid) > 0.19))
#===========================================================================


for f, tags in tqdm(df_test.values, miniters=1000):
    img = TEST_FOLDER + '{}.jpg'.format(f)
    x_test.append(img)

batch_size_test = 32
len_test = len(x_test)
x_tst = []
yfull_test = []


TTA_steps = 30

for k in range(0, TTA_steps):
    print(k)
    probs = model.predict_generator(predict_generator(x_test,IMG_SIZE,batch_size_test), steps=np.ceil(len(x_test)/batch_size_test),verbose=1)
    yfull_test.append(probs)
    k += 1

result = np.array(yfull_test[0])

for i in range(1, TTA_steps):
    result += np.array(yfull_test[i])
result /= TTA_steps

res = pd.DataFrame(result, columns=labels)
preds = []

for i in tqdm(range(res.shape[0]), miniters=1000):
    a = res.ix[[i]]
    a = a.apply(lambda x: x > X, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

print(len(preds))

df_test['tags'] = preds
df_test = df_test[:-57]
df_test.to_csv('submission.csv', index=False)
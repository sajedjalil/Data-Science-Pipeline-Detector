# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(2016)

import os
import glob
import cv2
import pickle
import datetime
import pandas as pd

from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss

# color type: 1 - grey, 3 - rgb
color_type_global = 1


def load_and_resize_image(path, img_rows, img_cols):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (img_cols, img_rows))


def get_driver_data():
    print('Read drivers data')
    path = os.path.join('..', 'input', 'driver_imgs_list.csv')
    df = pd.read_csv(path)      # Columns: subject, classname, img
    return df.set_index('img')['subject'].to_dict()


def load_train(img_rows, img_cols, color_type=1):
    X_train = []
    y_train = []
    driver_id = []

    driver_data = get_driver_data()

    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('..', 'input', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = load_and_resize_image(fl, img_rows, img_cols)
            X_train.append(img)
            y_train.append(j)
            driver_id.append(driver_data[flbase])

    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, driver_id, unique_drivers


def load_test(img_rows, img_cols):
    print('Read test images')
    path = os.path.join('..', 'input', 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = len(files)//10
    
    for fl in files:
        flbase = os.path.basename(fl)
        img = load_and_resize_image(fl, img_rows, img_cols)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total%thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id


def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    open(os.path.join('cache', 'architecture.json'), 'w').write(json_string)
    model.save_weights(os.path.join('cache', 'model_weights.h5'), overwrite=True)


def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, 
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 
                                                 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)


def read_and_normalize_train_data(img_rows, img_cols, color_type=1):
    train_data, train_target, driver_id, unique_drivers = load_train(img_rows, img_cols, color_type)
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    train_data = train_data.reshape(train_data.shape[0], color_type, img_rows, img_cols)
    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')
    train_data /= 255
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, driver_id, unique_drivers


def read_and_normalize_test_data(img_rows, img_cols, color_type=1):
    test_data, test_id = load_test(img_rows, img_cols)

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(test_data.shape[0], color_type, img_rows, img_cols)
    test_data = test_data.astype('float32')
    test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def copy_selected_drivers(train_data, train_target, driver_ids, driver_list):
    data = []
    target = []
    index = []
    for i, driver_id in enumerate(driver_ids):
        if driver_id in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    index = np.array(index, dtype=np.uint32)
    return data, target, index


def create_keras_model(img_rows, img_cols, channels=1):
    nb_classes = 10
    # number of convolutional filters to use
    nb_filters = 8
    # size of pooling area for max pooling
    nb_pool = 3
    # convolution kernel size
    nb_conv = 3
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            input_shape=(channels, img_rows, img_cols)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


if __name__ == '__main__':
    # Input image dimensions. 
    # Beware of the memory limit of 8192MB on Kaggle
    
    img_rows, img_cols = 24, 32
    batch_size = 32
    nb_epoch = 3
    random_state = 51

    train_data, train_target, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, 
                                                                                        color_type_global)
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type_global)

    yfull_train = dict()
    yfull_test = []
    unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                         'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                         'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                         'p075']
    X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, 
                                                          unique_list_train)
    unique_list_valid = ['p081']
    X_valid, Y_valid, test_indicies = copy_selected_drivers(train_data, train_target, driver_id,
                                                            unique_list_valid)

    print('Start Training')
    print('Split train: ', len(X_train), len(Y_train))
    print('Split valid: ', len(X_valid), len(Y_valid))
    print('Train drivers: ', unique_list_train)
    print('Test drivers: ', unique_list_valid)

    model = create_keras_model(img_rows, img_cols, color_type_global)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_valid, Y_valid),
              show_accuracy=True, 
              verbose=2)

    # score = model.evaluate(X_valid, Y_valid, show_accuracy=True, verbose=0)
    # print('Score log_loss: ', score[0])

    predictions_valid = model.predict(X_valid, batch_size=128, verbose=2)
    score = log_loss(Y_valid, predictions_valid)
    print('Score log_loss: ', score)

    # Store valid predictions
    for i, test_index in enumerate(test_indicies):
        yfull_train[test_index] = predictions_valid[i]

    # Store test predictions
    test_prediction = model.predict(test_data, batch_size=128, verbose=2)
    yfull_test.append(test_prediction)

    print('Final log_loss: {}, rows: {} cols: {} epoch: {}'.format(score, img_rows, img_cols, nb_epoch))
    info_string = 'loss_' + str(score) \
                    + '_r_' + str(img_rows) \
                    + '_c_' + str(img_cols) \
                    + '_ep_' + str(nb_epoch)

    test_res = merge_several_folds_mean(yfull_test, 1)
    create_submission(test_res, test_id, info_string)

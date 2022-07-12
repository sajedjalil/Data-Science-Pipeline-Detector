# -*- coding: utf-8 -*-
__author__ = 'Ravi: https://kaggle.com/company'

import numpy as np
np.random.seed(286)
import os
import glob
import cv2
import datetime
import time
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
z=0.49

def get_im_cv2(path, img_rows, img_cols):
    img = cv2.imread(path, 0)
    resized = cv2.resize(img, (img_cols, img_rows), cv2.INTER_LINEAR)
    return resized


def load_train(img_rows, img_cols):
    X_train = []
    X_train_id = []
    mask_train = []
    start_time = time.time()

    print('Read train images')
    files = glob.glob("../input/train/*[0-9].tif")
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, img_rows, img_cols)
        X_train.append(img)
        X_train_id.append(flbase[:-4])
        mask_path = "../input/train/" + flbase[:-4] + "_mask.tif"
        mask = get_im_cv2(mask_path, img_rows, img_cols)
        mask_train.append(mask)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, mask_train, X_train_id


def load_test(img_rows, img_cols):
    print('Read test images')
    files = glob.glob("../input/test/*[0-9].tif")
    X_test = []
    X_test_id = []
    total = 0
    start_time = time.time()
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, img_rows, img_cols)
        X_test.append(img)
        X_test_id.append(flbase[:-4])
        total += 1

    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test, X_test_id


def rle_encode(img, order='F'):
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []
    r = 0
    pos = 1
    for c in bytes:
        if c == 0:
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    if r != 0:
        runs.append((pos, r))
        pos += r

    z = ''
    for rr in runs:
        z += str(rr[0]) + ' ' + str(rr[1]) + ' '
    return z[:-1]


def find_best_mask():
    files = glob.glob(os.path.join("..", "input", "train", "*_mask.tif"))
    overall_mask = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    overall_mask.fill(0)
    overall_mask = overall_mask.astype(np.float32)

    for fl in files:
        mask = cv2.imread(fl, cv2.IMREAD_GRAYSCALE)
        overall_mask += mask
    overall_mask /= 255
    max_value = overall_mask.max()
    koeff = z #0.5
    overall_mask[overall_mask < koeff * max_value] = 0
    overall_mask[overall_mask >= koeff * max_value] = 255
    overall_mask = overall_mask.astype(np.uint8)
    return overall_mask


def create_submission(predictions, test_id, info):
    sub_file = os.path.join('submission_v3_0.49' + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
    subm = open(sub_file, "w")
    mask = find_best_mask()
    encode = rle_encode(mask)
    subm.write("img,pixels\n")
    for i in range(len(test_id)):
        subm.write(str(test_id[i]) + ',')
        if predictions[i][1] > z:
            subm.write(encode)
        subm.write('\n')
    subm.close()


def get_empty_mask_state(mask):
    out = []
    for i in range(len(mask)):
        if mask[i].sum() == 0:
            out.append(0)
        else:
            out.append(1)
    return np.array(out)


def read_and_normalize_train_data(img_rows, img_cols):
    train_data, train_target, train_id = load_train(img_rows, img_cols)
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
    # Convert to 0 or 1
    train_target = get_empty_mask_state(train_target)
    train_target = np_utils.to_categorical(train_target, 2)
    train_data = train_data.astype('float32')
    train_data /= 255
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data(img_rows, img_cols):
    test_data, test_id = load_test(img_rows, img_cols)
    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
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


def create_model(img_rows, img_cols):
    model = Sequential()
    model.add(Convolution2D(4, 4, 4, border_mode='same', init='he_normal',
                            input_shape=(1, img_rows, img_cols)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.508))

    model.add(Convolution2D(8, 4, 4, border_mode='same', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.608))

    model.add(Flatten())
    model.add(Dense(2))
    model.add(Activation('softmax'))

    sgd = SGD(lr=1e-3, decay=.8e-6, momentum=0.901, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def getPredScorePercent(train_target, train_id, predictions_valid):
    perc = 0
    for i in range(len(train_target)):
        pred = 1
        if predictions_valid[i][0] > z:
            pred = 0
        real = 1
        if train_target[i][0] > z:
            real = 0
        if real == pred:
            perc += 1
    perc /= len(train_target)
    return perc


def run_cross_validation(nfolds=10):
    img_rows, img_cols = 32, 32
    batch_size = 32
    nb_epoch = 100
    random_state = 51

    train_data, train_target, train_id = read_and_normalize_train_data(img_rows, img_cols)
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols)

    yfull_train = dict()
    yfull_test = []
    kf = KFold(len(train_data), n_folds=nfolds,  shuffle=True,random_state=random_state)
    num_fold = 0
    sum_score = 0
    for train_index, test_index in kf:
        model = create_model(img_rows, img_cols)
        X_train, X_valid = train_data[train_index], train_data[test_index]
        Y_train, Y_valid = train_target[train_index], train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, verbose=0),
        ]
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
              callbacks=callbacks)

        predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        # Store test predictions
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    predictions_valid = get_validation_predictions(train_data, yfull_train)
    score = log_loss(train_target, predictions_valid)
    print("Log_loss train independent avg: ", score)

    print('Final log_loss: {}, rows: {} cols: {} nfolds: {} epoch: {}'.format(score, img_rows, img_cols, nfolds, nb_epoch))
    perc = getPredScorePercent(train_target, train_id, predictions_valid)
    print('Percent success: {}'.format(perc))

    info_string = 'loss_' + str(score) \
                    + '_r_' + str(img_rows) \
                    + '_c_' + str(img_cols) \
                    + '_folds_' + str(nfolds) \
                    + '_ep_' + str(nb_epoch)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    create_submission(test_res, test_id, info_string)


if __name__ == '__main__':
    run_cross_validation(5)

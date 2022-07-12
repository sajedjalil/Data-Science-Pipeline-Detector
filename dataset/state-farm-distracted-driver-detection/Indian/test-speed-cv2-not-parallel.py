# Test which library read and resize JPG files faster
# cv2 vs scipy

import numpy as np
np.random.seed(2016)

import os
import glob
import cv2
import math
import time
from scipy.misc import imread, imresize
import tensorflow as tf
from PIL import image


def get_im_skipy(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = imread(path, True)
    elif color_type == 3:
        img = imread(path)
    # Reduce size
    resized = imresize(img, (img_cols, img_rows))
    return resized


def get_im_cv2(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows)).transpose((2,0,1)).astype('float32') / 255.0
    return resized


def get_im_tf(path, img_rows, img_cols, color_type=1):
    img = tf.read_file(path)
    # Load as grayscale
    if color_type == 1:
        img = tf.image.decode_jpeg(img, channels=1)
    elif color_type == 3:
        img = tf.image.decode_jpeg(img, channels=0)
    # Reduce size
    resized = tf.image.resize_images(img, img_rows, img_cols, method=0)
    return resized


def get_driver_data():
    dr = dict()
    path = os.path.join('..', 'input', 'driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr


def load_train(img_rows, img_cols, color_type=1, type=0):
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
            if type == 0:
                img = get_im_cv2(fl, img_rows, img_cols, color_type)
            elif type == 1:
                img = get_im_skipy(fl, img_rows, img_cols, color_type)
            else:
                img = get_im_tf(fl, img_rows, img_cols, color_type)
            X_train.append(img)
            y_train.append(j)
            driver_id.append(driver_data[flbase])

    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, driver_id, unique_drivers


def load_test(img_rows, img_cols, color_type=1, type=0):
    print('Read test images')
    path = os.path.join('..', 'input', 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        if type == 0:
            img = get_im_cv2(fl, img_rows, img_cols, color_type)
        elif type == 1:
            img = get_im_skipy(fl, img_rows, img_cols, color_type)
        else:
            img = get_im_tf(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total%thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id


start_time = time.time()
load_train(50, 50, 3, 0)
load_test(50, 50, 3, 0)
cv2_time = time.time() - start_time
print(cv2_time, ' seconds')
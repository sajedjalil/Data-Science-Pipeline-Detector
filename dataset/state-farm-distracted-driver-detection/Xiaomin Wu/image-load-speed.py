import numpy as np
import os
import time
import cv2
import tensorflow as tf
from skimage import io

def load_local_train_val():
    train_path = 'path/to/train'
    labels = []
    img_arrs = []
    for c in os.listdir(train_path):
        begin = time.time()
        flag = 0
        train_path_c = train_path + '/' + c
        label = int(c[-1])
        for img in os.listdir(train_path_c):
            img_path = train_path_c + '/' +img
            #img_arr = cv2.imread(img_path)
            #img_arr = io.imread(img_path)
            img_arr = tf.image.decode_jpeg(img_path)
            labels.append(label)
            img_arrs.append(img_arr)
            flag += 1
        print flag
        end = time.time()
        print 'time: {0}'.format(end-begin)
    print np.asarray(labels).shape
    print np.asarray(img_arrs).shape

if __name__ == '__main__':
    load_local_train_val()
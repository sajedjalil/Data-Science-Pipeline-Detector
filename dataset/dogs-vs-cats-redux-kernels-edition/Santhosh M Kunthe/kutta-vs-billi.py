import pandas as pd
import tensorflow as tf
import numpy as np
import zipfile
import cv2
import matplotlib.pyplot as plt
import os

TRAIN_ZIP = '../input/dogs-vs-cats-redux-kernels-edition/train.zip'
zip_ref = zipfile.ZipFile(TRAIN_ZIP, 'r')
zip_ref.extractall('training')
zip_ref.close()

TRAIN_DIR = '/kaggle/working/training/train/'

labelList = []
for i in os.listdir(TRAIN_DIR):
    # print(i)
    labelList.append(i)

import random

random.shuffle(labelList)
labelList = labelList[0:15000]

initializer = tf.initializers.glorot_uniform()

optimizer = tf.optimizers.Adam(learning_rate=0.001)
tf.random.set_seed(500)

m1 = tf.Variable(initializer([3, 3, 3, 256]), trainable=True, dtype=tf.float32)
b1 = tf.Variable(initializer([256]), trainable=True, dtype=tf.float32)

m2 = tf.Variable(initializer([3, 3, 256, 128]), trainable=True, dtype=tf.float32)
b2 = tf.Variable(initializer([128]), trainable=True, dtype=tf.float32)

m3 = tf.Variable(initializer([3, 3, 128, 32]), trainable=True, dtype=tf.float32)
b3 = tf.Variable(initializer([32]), trainable=True, dtype=tf.float32)

m4 = tf.Variable(initializer([3, 3, 32, 8]), trainable=True, dtype=tf.float32)
b4 = tf.Variable(initializer([8]), trainable=True, dtype=tf.float32)

m5 = tf.Variable(initializer([3, 3, 8, 3]), trainable=True, dtype=tf.float32)
b5 = tf.Variable(initializer([3]), trainable=True, dtype=tf.float32)

m6 = tf.Variable(initializer([507, 128]), trainable=True, dtype=tf.float32)
b6 = tf.Variable(initializer([128]), trainable=True, dtype=tf.float32)

m7 = tf.Variable(initializer([128, 32]), trainable=True, dtype=tf.float32)
b7 = tf.Variable(initializer([32]), trainable=True, dtype=tf.float32)

m8 = tf.Variable(initializer([32, 2]), trainable=True, dtype=tf.float32)
b8 = tf.Variable(initializer([2]), trainable=True, dtype=tf.float32)

lossFactor = 5000
leastLoss = 5000
lossThreshould = 6.93 #5  # 7.2
seenWholeDataTimes = 0

while lossFactor > lossThreshould or seenWholeDataTimes < 3:
    seenWholeDataTimes = seenWholeDataTimes + 1

    for i in range((int)(labelList.__len__() / 10)):
        # print(i)
        if lossFactor < lossThreshould and seenWholeDataTimes > 3:
            break
        with tf.GradientTape() as tape:
            miniLabelList = labelList[(i * 10): ((i + 1) * 10)]
            miniImageList = []
            miniLabelHotArray = []

            for j in range(miniLabelList.__len__()):
                readedImage = cv2.imread(TRAIN_DIR + miniLabelList[j])
                readedImage = cv2.resize(readedImage, (405, 405), 0, 0, cv2.INTER_LINEAR)
                # print(readedImage.shape)
                # plt.imshow(readedImage)
                # plt.show()
                miniImageList.append(readedImage)
                if 'dog' in miniLabelList[j]:
                    miniLabelHotArray.append([1, 0])
                elif 'cat' in miniLabelList[j]:
                    miniLabelHotArray.append([0, 1])
                else:
                    miniLabelHotArray.append([0, 0])

            miniImageList = tf.cast(miniImageList, tf.float32)
            miniLabelHotArray = tf.cast(miniLabelHotArray, tf.float32)
            # print(miniImageList.shape, miniLabelHotArray.shape)

            mx_ba = tf.nn.conv2d(input=miniImageList, filters=m1, strides=[1, 1, 1, 1], padding='SAME')
            mx_ba = tf.nn.bias_add(mx_ba, b1)
            mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            mx_ba = tf.nn.conv2d(input=mx_ba, filters=m2, strides=[1, 1, 1, 1], padding='SAME')
            mx_ba = tf.nn.bias_add(mx_ba, b2)
            mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            mx_ba = tf.nn.conv2d(input=mx_ba, filters=m3, strides=[1, 1, 1, 1], padding='SAME')
            mx_ba = tf.nn.relu(tf.nn.bias_add(mx_ba, b3))
            mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            mx_ba = tf.nn.conv2d(input=mx_ba, filters=m4, strides=[1, 1, 1, 1], padding='SAME')
            mx_ba = tf.nn.bias_add(mx_ba, b4)
            mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            mx_ba = tf.nn.conv2d(input=mx_ba, filters=m5, strides=[1, 1, 1, 1], padding='SAME')
            mx_ba = tf.nn.relu(tf.nn.bias_add(mx_ba, b5))
            mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            print(mx_ba.shape,"-")

            mx_baTemp = []
            for m in range(10):
                mx_baTemp.append(tf.reshape(mx_ba[m], [507]))
            mx_ba = tf.cast(mx_baTemp, tf.float32)
            # print(mx_ba.shape,"*")

            mx_ba = tf.add(tf.matmul(mx_ba, m6), b6)
            mx_ba = tf.nn.relu(tf.add(tf.matmul(mx_ba, m7), b7))
            mx_ba = tf.add(tf.matmul(mx_ba, m8), b8)

            loss = tf.nn.softmax_cross_entropy_with_logits(miniLabelHotArray, mx_ba, axis=-1)
            lossFactor = loss.numpy().sum(axis=0)
            if leastLoss > lossFactor:
                leastLoss = lossFactor
            print(seenWholeDataTimes, "--", leastLoss, "--", i, "--", lossFactor, "--", mx_ba.shape, "--",
                  miniLabelHotArray.shape)
            mx_baIgnore = tf.nn.relu(mx_ba)
            # print(loss)
            '''
            for m in range(10):
                if (mx_baIgnore[m].numpy() == miniLabelHotArray[m].numpy()).all():
                    print("--")
                else:
                    print("**************")
            '''
            tv = [m1, b1, m3, b3, m4, b4, m6, b6, m7, b7, m8, b8]
            grads = tape.gradient(loss, tv)
            optimizer.apply_gradients(zip(grads, tv))

TEST_ZIP = '../input/dogs-vs-cats-redux-kernels-edition/test.zip'
zip_ref = zipfile.ZipFile(TEST_ZIP, 'r')
zip_ref.extractall('testing')
zip_ref.close()

TEST_DIR = '/kaggle/working/testing/test/'

testNameList = []
predList = []

for i in os.listdir(TEST_DIR):
    # print(i)
    testNameList.append(i)
#testNameList = testNameList[0:200]
for i in range((int)(testNameList.__len__() / 10)):

    miniLabelList = testNameList[(i * 10): ((i + 1) * 10)]
    miniImageList = []
    for j in range(miniLabelList.__len__()):
        readedImage = cv2.imread(TEST_DIR + miniLabelList[j])
        readedImage = cv2.resize(readedImage, (405, 405), 0, 0, cv2.INTER_LINEAR)
        # print(readedImage.shape)
        miniImageList.append(readedImage)

    miniImageList = tf.cast(miniImageList, tf.float32)

    mx_ba = tf.nn.conv2d(input=miniImageList, filters=m1, strides=[1, 1, 1, 1], padding='SAME')
    mx_ba = tf.nn.bias_add(mx_ba, b1)
    mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    mx_ba = tf.nn.conv2d(input=mx_ba, filters=m2, strides=[1, 1, 1, 1], padding='SAME')
    mx_ba = tf.nn.bias_add(mx_ba, b2)
    mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    mx_ba = tf.nn.conv2d(input=mx_ba, filters=m3, strides=[1, 1, 1, 1], padding='SAME')
    mx_ba = tf.nn.relu(tf.nn.bias_add(mx_ba, b3))
    mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    mx_ba = tf.nn.conv2d(input=mx_ba, filters=m4, strides=[1, 1, 1, 1], padding='SAME')
    mx_ba = tf.nn.bias_add(mx_ba, b4)
    mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    mx_ba = tf.nn.conv2d(input=mx_ba, filters=m5, strides=[1, 1, 1, 1], padding='SAME')
    mx_ba = tf.nn.relu(tf.nn.bias_add(mx_ba, b5))
    mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    print(mx_ba.shape,"-")

    mx_baTemp = []
    for m in range(10):
        mx_baTemp.append(tf.reshape(mx_ba[m], [507]))
    mx_ba = tf.cast(mx_baTemp, tf.float32)
    # print(mx_ba.shape,"*")

    mx_ba = tf.add(tf.matmul(mx_ba, m6), b6)
    mx_ba = tf.nn.relu(tf.add(tf.matmul(mx_ba, m7), b7))
    mx_ba = tf.add(tf.matmul(mx_ba, m8), b8)

    for k in range(10):
        # print(mx_ba[k].numpy())
        # print(mx_ba[k][0] ,"-*-*-", mx_ba[k][1])
        if (mx_ba[k][0] > mx_ba[k][1]):
            predList.append(1)
            print(i, "-/-", k, "-/- dog")
        else:
            predList.append(0)
            print(i, "-/-", k, "-/- cat")

subdf = pd.DataFrame()

subdf['id'] = testNameList
subdf['label'] = predList

subdf.to_csv('submission.csv', index=False)

print(subdf)

trainingFilePath = "/kaggle/input/plant-pathology-2020-fgvc7/images/"

import pandas as pd
import tensorflow as tf
import numpy as np
import cv2

optimizer = tf.optimizers.Adam(learning_rate=0.001)
tf.random.set_seed(500)

completeTrainingData = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
# print(len(completeTrainingData))
from sklearn.utils import shuffle

# completeTrainingData = shuffle(completeTrainingData)
imageNameList = completeTrainingData['image_id']
# print(imageNameList.size)
completeLabelList = []
for i in range(imageNameList.__len__()):
    completeLabelList.append([(completeTrainingData['healthy'][i]) * 100,
                              (completeTrainingData['multiple_diseases'][i]) * 100,
                              (completeTrainingData['rust'][i]) * 100,
                              (completeTrainingData['scab'][i]) * 100
                              ])

m1 = tf.Variable(tf.random.normal([3, 3, 3, 32], .00009, .00009, tf.float32, seed=1))
m2 = tf.Variable(tf.random.normal([3, 3, 32, 64], .00009, .00009, tf.float32, seed=1))
m3 = tf.Variable(tf.random.normal([3, 3, 64, 128], .00009, .00009, tf.float32, seed=1))
m4 = tf.Variable(tf.random.normal([3, 3, 128, 3], .00009, .00009, tf.float32, seed=1))
m5 = tf.Variable(tf.random.normal([3, 3, 3, 3], .00009, .00009, tf.float32, seed=1))

m6 = tf.Variable(tf.random.normal([12, 4], .00009, .00009, tf.float32, seed=1))
b6 = tf.Variable(tf.random.normal([4], .00009, .00009, tf.float32, seed=1))

# print(completeTrainingData[0:5])
# print(completeTrainingData.head())
lossFactor = 5000
lossThreshould = 1710
while lossFactor > lossThreshould:
    for i in range((int)(completeLabelList.__len__() / 10)):
        if lossFactor < lossThreshould:
            break
        with tf.GradientTape() as tape:
            if lossFactor < lossThreshould:
                break
            miniBatch = completeTrainingData[(i * 10):((i + 1) * 10)]
            # print("\n\n***********************\n---->",i,"\n",miniBatch)
            # print(imageNameList[(i*10):((i+1)*10)])
            # print("\n########\n")
            # print(completeLabelList[(i*10):((i+1)*10)])
            # print("\n###############################################\n")

            miniImageSet = imageNameList[(i * 10):((i + 1) * 10)]
            miniLabelList = completeLabelList[(i * 10):((i + 1) * 10)]

            featureImageList = []
            for eachImage in miniImageSet:
                if lossFactor < lossThreshould:
                    break
                readedImage = cv2.imread(trainingFilePath + eachImage + ".jpg")
                readedImage = tf.cast(readedImage, tf.float32)
                readedImage = tf.reshape(readedImage, [1365, 2048, 3])
                # print(readedImage.shape)
                featureImageList.append(readedImage)
            featureImageList = tf.cast(featureImageList, tf.float32)
            miniLabelList = tf.cast(miniLabelList, tf.float32)
            # print(featureImageList.shape)

            mx_ba = tf.nn.conv2d(input=featureImageList, filters=m1, strides=[1, 1, 1, 1], padding='SAME')
            mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            mx_ba = tf.nn.conv2d(input=mx_ba, filters=m2, strides=[1, 1, 1, 1], padding='SAME')
            mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            mx_ba = tf.nn.conv2d(input=mx_ba, filters=m3, strides=[1, 1, 1, 1], padding='SAME')
            mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
            mx_ba = tf.nn.relu(mx_ba)

            mx_ba = tf.nn.conv2d(input=mx_ba, filters=m4, strides=[1, 1, 1, 1], padding='SAME')
            mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

            mx_ba = tf.nn.conv2d(input=mx_ba, filters=m5, strides=[1, 1, 1, 1], padding='SAME')
            mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
            mx_ba = tf.nn.relu(mx_ba)

            mx_ba = tf.reshape(mx_ba, [10, 12])

            mx_ba = tf.add(tf.matmul(mx_ba, m6), b6)

            loss = tf.math.reduce_mean(tf.math.square(mx_ba - miniLabelList) + 1)
            grads = tape.gradient(loss, [m1, m2, m3, m4, m5, m6, b6])
            optimizer.apply_gradients(zip(grads, [m1, m2, m3, m4, m5, m6, b6]))
            lossFactor = loss.numpy()
            print(i, ' == ', miniLabelList.shape, '===', mx_ba.shape, " --**-- ", lossFactor)

completeTestingData = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')
testNameList = completeTestingData['image_id']
testHealthy = []
testMulDis = []
testRust = []
testScab = []

for i in range((int)(testNameList.__len__() / 10)):

    miniImageSet = testNameList[(i * 10):((i + 1) * 10)]

    featureImageList = []
    for eachImage in miniImageSet:
        readedImage = cv2.imread(trainingFilePath + eachImage + ".jpg")
        readedImage = tf.cast(readedImage, tf.float32)
        readedImage = tf.reshape(readedImage, [1365, 2048, 3])
        # print(readedImage.shape)
        featureImageList.append(readedImage)
    featureImageList = tf.cast(featureImageList, tf.float32)

    mx_ba = tf.nn.conv2d(input=featureImageList, filters=m1, strides=[1, 1, 1, 1], padding='SAME')
    mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    mx_ba = tf.nn.conv2d(input=mx_ba, filters=m2, strides=[1, 1, 1, 1], padding='SAME')
    mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    mx_ba = tf.nn.conv2d(input=mx_ba, filters=m3, strides=[1, 1, 1, 1], padding='SAME')
    mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    mx_ba = tf.nn.relu(mx_ba)

    mx_ba = tf.nn.conv2d(input=mx_ba, filters=m4, strides=[1, 1, 1, 1], padding='SAME')
    mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

    mx_ba = tf.nn.conv2d(input=mx_ba, filters=m5, strides=[1, 1, 1, 1], padding='SAME')
    mx_ba = tf.nn.max_pool(input=mx_ba, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    mx_ba = tf.nn.relu(mx_ba)

    mx_ba = tf.reshape(mx_ba, [10, 12])

    mx_ba = tf.add(tf.matmul(mx_ba, m6), b6)

    for j in range(10):
        tempSubArr = [0, 0, 0, 0]
        tempSubArr[np.argmax(mx_ba[j], axis=0)] = 1

        testHealthy.append(tempSubArr[0])
        testMulDis.append(tempSubArr[1])
        testRust.append(tempSubArr[2])
        testScab.append(tempSubArr[3])
        #print(i, "--", j, "---", mx_ba[j].numpy(), "-*-", tempSubArr)

testHealthy.append(0)
testMulDis.append(0)
testRust.append(0)
testScab.append(1)

subdf = pd.DataFrame()

subdf['image_id'] = testNameList
subdf['healthy'] = testHealthy
subdf['multiple_diseases'] = testMulDis
subdf['rust'] = testRust
subdf['scab'] = testScab

subdf.to_csv('submission.csv', index=False)

print(subdf)




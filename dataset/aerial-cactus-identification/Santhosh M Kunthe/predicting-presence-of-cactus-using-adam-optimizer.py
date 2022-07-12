cactusTrainFilePath = '../input/train/train/'

import pandas as pd

featureLabelCSV = pd.read_csv(
    '../input/train.csv')
imageListFromCSV = featureLabelCSV['id']
labelListFromCSV = featureLabelCSV['has_cactus']

featureList = []
labelList = []

for eachId in imageListFromCSV:
    import cv2

    tempImage = cv2.imread(cactusTrainFilePath + eachId)
    featureList.append(tempImage)

import numpy as np

featureListNP = np.array(featureList)
labelListNP = np.array(labelListFromCSV)
labelListNP = np.reshape(labelListNP, [-1, 1])
print(featureListNP.shape)
print(labelListNP.shape)

import tensorflow as tf

y = tf.placeholder(tf.float32, shape=[None, 1])

mForLayer1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 32], stddev=0.05))
mForLayer2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], stddev=0.05))
mForLayer3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], stddev=0.05))

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])

bForLayer1 = tf.Variable(tf.constant(0.05, shape=[32]))
bForLayer2 = tf.Variable(tf.constant(0.05, shape=[32]))
bForLayer3 = tf.Variable(tf.constant(0.05, shape=[64]))

mx1 = tf.nn.conv2d(input=x, filter=mForLayer1, strides=[1, 1, 1, 1], padding='SAME')
mx_b1 = mx1 + bForLayer1
mx_b1Pooled = tf.nn.max_pool(value=mx_b1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

mx2 = tf.nn.conv2d(input=mx_b1Pooled, filter=mForLayer2, strides=[1, 1, 1, 1], padding='SAME')
mx_b2 = mx2 + bForLayer2
mx_b2Pooled = tf.nn.max_pool(value=mx_b2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
mx_b2PooledActivated = tf.nn.relu(mx_b2Pooled)

mx3 = tf.nn.conv2d(input=mx_b2PooledActivated, filter=mForLayer3, strides=[1, 1, 1, 1], padding='SAME')
mx_b3 = mx3 + bForLayer3
mx_b3Pooled = tf.nn.max_pool(value=mx_b3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# IMAGE TO LABEL SHAPE
layer_shape = mx_b3Pooled.get_shape()
num_features = layer_shape[1:4].num_elements()
mx_b3Pooled = tf.reshape(mx_b3Pooled, [-1, num_features])

mForFinalLayer1 = tf.Variable(
    tf.truncated_normal(shape=[mx_b3Pooled.get_shape()[1:4].num_elements(), 32], stddev=0.05))
bForFinalLayer1 = tf.Variable(tf.constant(0.05, shape=[32]))

mx4 = tf.matmul(mx_b3Pooled, mForFinalLayer1)
mx4Activated = tf.nn.relu(mx4)

mForFinalLayer2 = tf.Variable(tf.truncated_normal(shape=[32, 1], stddev=0.05))
bForFinalLayer2 = tf.Variable(tf.constant(0.05, shape=[1]))

mx5 = tf.matmul(mx4Activated, mForFinalLayer2) + bForFinalLayer2

session = tf.Session()

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mx5, labels=y))
error = tf.add(mx5, -y)
trainingStep = tf.train.AdamOptimizer(learning_rate=.0001).minimize(loss)
session.run(tf.global_variables_initializer())

print(session.run(mx5, feed_dict={x: featureListNP}))

for i in range(1332):
    session.run([trainingStep], feed_dict={x: featureListNP, y: labelListNP})
    # print(np.sum(np.array(session.run([error], feed_dict={x: featureListNP, y: labelListNP}))))
    # print(i)
    if i % 100 == 0:
        print(np.sum(np.array(session.run([error], feed_dict={x: featureListNP, y: labelListNP}))))
        print(i)

import os

testImageList = os.listdir('../input/test/test')
print(testImageList)
testFeatureList = []
for eachtestImagePath in testImageList:
    import cv2

    imageTepm = cv2.imread(
        '../input/test/test/' + eachtestImagePath)
    testFeatureList.append(imageTepm)

testFeatureListNp = np.array(testFeatureList)
print(testFeatureListNp.shape)

predictionPropabilityList = np.array(session.run(tf.nn.relu(mx5), feed_dict={x: testFeatureListNp}))
predictionPropabilityList[np.isnan(predictionPropabilityList)] = 0
predictionSum = np.sum(predictionPropabilityList)
predAvg = predictionSum / len(predictionPropabilityList)

print(predAvg, predictionSum, predictionPropabilityList)

prediction0or1 = []

for eachPredictionPropabilityList in predictionPropabilityList:
    if eachPredictionPropabilityList > predAvg:
        prediction0or1.append(1)
    else:
        prediction0or1.append(0)

for i in range(len(prediction0or1)):
    print('Status of ' + testImageList[i] + ' having cactus is ' + prediction0or1[i].__str__())

outputDF = pd.DataFrame()
outputDF['id'] = testImageList
outputDF['has_cactus'] = prediction0or1
outputDF = outputDF.set_index('id')
outputDF.to_csv('submission.csv')
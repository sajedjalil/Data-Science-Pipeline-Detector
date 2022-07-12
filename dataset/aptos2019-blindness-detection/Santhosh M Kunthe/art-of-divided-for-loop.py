trainingFilePath = '../input/train_images/'

import pandas as pd

trainingCSVdata = pd.read_csv(
    '../input/train.csv')
# print(trainingCSVdata.head())

imageList = trainingCSVdata['id_code']
inputSize = len(imageList)
inputSize = int(inputSize / 10)
inputSize = inputSize * 10
imageList = imageList[0:inputSize]
featureImageList = []
for eachImage in imageList:
    import cv2

    tempImage = cv2.imread(trainingFilePath + eachImage + '.png')
    tempImage = cv2.resize(tempImage, (224, 224), 0, 0, cv2.INTER_LINEAR)
    featureImageList.append(tempImage)
    # print(len(featureImageList))
featureImageList = featureImageList[0:inputSize]

labelList = trainingCSVdata['diagnosis']
labelList = labelList[0:inputSize]

labelListHotArray = []
for eachLabel in labelList:
    # print(eachLabel)
    #[(0, 1805), (2, 999), (1, 370), (4, 295), (3, 193)]
    # 9--2--5--1--2
    if eachLabel == 0:
        labelListHotArray.append([3, 1, 0, 2, 2])
    elif eachLabel == 1:
        labelListHotArray.append([0, 6, 0, 0, 0])
    elif eachLabel == 2:
        labelListHotArray.append([0, 0, 5, 1, 1])
    elif eachLabel == 3:
        labelListHotArray.append([0, 0, 0, 6, 0])
    elif eachLabel == 4:
        labelListHotArray.append([0, 0, 0, 0, 6])
    else:
        labelListHotArray.append([0, 0, 0, 0, 0])

import numpy as np

featureNP = np.array(featureImageList)
labelNP = np.array(labelListHotArray)
# labelNP = np.reshape(labelNP, [-1, 5])
# print(featureNP.shape, labelNP.shape)

featureImageList = []
imageList = []
import tensorflow as tf

y = tf.placeholder(tf.float32, shape=[None, 5])

m1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=0.05))
m2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 32], stddev=0.05))
m3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 1], stddev=0.05))

x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])

b1 = tf.Variable(tf.constant(0.05, shape=[64]))
b2 = tf.Variable(tf.constant(0.05, shape=[32]))
b3 = tf.Variable(tf.constant(0.05, shape=[1]))

mx_b = tf.nn.conv2d(input=x, filter=m1, strides=[1, 1, 1, 1], padding='SAME') + b1
mx_b = tf.nn.max_pool(value=mx_b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
mx_b = tf.nn.relu(mx_b)

mx_b = tf.nn.conv2d(input=mx_b, filter=m2, strides=[1, 1, 1, 1], padding='SAME') + b2
mx_b = tf.nn.max_pool(value=mx_b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
mx_b = tf.nn.relu(mx_b)

mx_b = tf.nn.conv2d(input=mx_b, filter=m3, strides=[1, 1, 1, 1], padding='SAME') + b3
mx_b = tf.nn.max_pool(value=mx_b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
mx_b = tf.nn.relu(mx_b)
# -------------------------------------


# IMAGE TO LABEL SHAPE
layer_shape = mx_b.get_shape()
num_features = layer_shape[1:4].num_elements()
mx_b3PooledActivated = tf.reshape(mx_b, [-1, num_features])

mForFinalLayer1 = tf.Variable(
    tf.truncated_normal(shape=[mx_b3PooledActivated.get_shape()[1:4].num_elements(), 128], stddev=0.05))
bForFinalLayer1 = tf.Variable(tf.constant(0.05, shape=[128]))

mx4 = tf.matmul(mx_b3PooledActivated, mForFinalLayer1)
mx4Activated = tf.nn.relu(mx4)

mForFinalLayer2 = tf.Variable(tf.truncated_normal(shape=[128, 5], stddev=0.05))
bForFinalLayer2 = tf.Variable(tf.constant(0.05, shape=[5]))

mx5 = tf.matmul(mx4Activated, mForFinalLayer2) + bForFinalLayer2

# ------------------------------------------
sess = tf.Session()

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mx5, labels=y))
error = tf.reduce_mean(tf.sqrt(tf.square(mx5 - y)))
trainingStep = tf.train.AdamOptimizer(learning_rate=.0001).minimize(loss)

sess.run(tf.global_variables_initializer())

for i in range(int(inputSize / 10) - 1):
    for j in range(3):
        sess.run([trainingStep, loss, error],
                 feed_dict={x: featureNP[(i * 10):((i + 1) * 10)], y: labelNP[(i * 10):((i + 1) * 10)]})
        # print(i, j, "1")

for i in range(int(inputSize / 10) - 1):
    for j in range(3):
        sess.run([trainingStep, loss, error],
                 feed_dict={x: featureNP[(i * 10):((i + 1) * 10)], y: labelNP[(i * 10):((i + 1) * 10)]})
        # print(i, j, "2")

for i in range(int(inputSize / 10) - 1):
    for j in range(3):
        sess.run([trainingStep, loss, error],
                 feed_dict={x: featureNP[(i * 10):((i + 1) * 10)], y: labelNP[(i * 10):((i + 1) * 10)]})
        # print(i,
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
featureNP = np.array([0])
testingFilePath = '../input/test_images/'
testingCSVdata = pd.read_csv(
    '../input/test.csv')
# print(testingCSVdata.head())

imageList = testingCSVdata['id_code']
# imageList = imageList[0:20]
testfeatureImageList = []
for eachImage in imageList:
    import cv2

    tempImage = cv2.imread(testingFilePath + eachImage + '.png')
    tempImage = cv2.resize(tempImage, (224, 224), 0, 0, cv2.INTER_LINEAR)
    testfeatureImageList.append(tempImage)
    # print(len(testfeatureImageList))

lengthOfTestImages = len(testfeatureImageList)
testFeatureNP = np.array(testfeatureImageList)
testfeatureImageList = []

lengthOfTestImagesONETH = lengthOfTestImages % 10
lengthOfTestImagesTENTH = lengthOfTestImages - lengthOfTestImagesONETH
noOfForLoops = lengthOfTestImagesTENTH / 10

outPutTotal = []
for i in range(int(lengthOfTestImages)):
    outputTemp = np.array(sess.run([mx5], feed_dict={x: testFeatureNP[i:(i + 1)]})).tolist()
    outPutTotal.append(outputTemp)
    # print(i)

output = np.array(outPutTotal)

# print(output.shape)
outputDF = pd.DataFrame()
labelListOutput = []
for eachOutput in output:
    # print(eachOutput)
    # print(np.argmax(eachOutput))
    labelListOutput.append(np.argmax(eachOutput))

# print(np.array(imageList).shape, np.array(labelListOutput).shape, len(imageList), len(labelListOutput))

outputDF['id_code'] = imageList
outputDF['diagnosis'] = labelListOutput

outputDF.to_csv('submission.csv', index=False)

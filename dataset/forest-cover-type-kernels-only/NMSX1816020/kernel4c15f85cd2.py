# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
path = "../input/"

# Any results you write to the current directory are saved as output.
import tensorflow as tf

filepath = "../input/"
INPUT_NODE = 54
OUTPUT_NODE = 7

REGULARIZATION_RATE = 0.0001
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

TRAINING_STEP = 150000

def processLabel(labels):
    t = np.zeros(shape=[labels.shape[0],7])
    for i in range(labels.shape[0]):
        t[i, int(labels[i])-1] = 1.0
    return t


def readDataSet(filename):
    file = np.loadtxt(filename,dtype=np.float32,delimiter=",",skiprows=1)
    features = file[0:, 1:55]
    labels = file[0:, 55:]
    labels_std = processLabel(labels)
    return [features, labels_std]

def readTestSet(filename):
    file = np.loadtxt(filename, dtype=np.float32, delimiter=",", skiprows=1)
    feature = file[0:,1:55]
    return feature

def main(argv=None):
    trainFeature, trainLabel = readDataSet(filepath + "/train.csv")
    testFeature = readTestSet(filepath + "/test.csv")

    print(trainFeature.shape,trainFeature.shape)
    train(trainFeature, trainLabel, testFeature)

def forward(input_tensor, weight, bias):
    return tf.matmul(input_tensor, weight) + bias

def train(features, labels, testFeature):
    x = tf.placeholder(tf.float32, shape=[None, features.shape[1]], name="input")
    y_ = tf.placeholder(tf.float32, shape=[None, labels.shape[1]], name="y-input")

    Id = np.linspace(features.shape[0]+1, features.shape[0] + testFeature.shape[0], testFeature.shape[0])

    weight = tf.Variable(tf.truncated_normal([INPUT_NODE, OUTPUT_NODE], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    
    y = forward(x, weight, bias)

    global_step = tf.Variable(0, trainable=False)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weight)
    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, features.shape[0] / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    correct_predection = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predection, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x:features[14000:,:], y_:labels[14000:]}
        train_feature = features[0:14000,:]
        train_lable = labels[0:14000]

        test_feed = {x:testFeature}
        for i in range(TRAINING_STEP):
            if i%1000==0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("%d now the vali_acc is %g" %(i, validate_acc))
            xs = train_feature[i%140*100:i%140*100+100, :]
            ys = train_lable[i%140*100:i%140*100+100]
            sess.run(train_step, feed_dict={x:xs, y_:ys})
        validate_acc = sess.run(accuracy, feed_dict=validate_feed)
        print("After training the vali_acc is %g" %validate_acc)
        outputTenor = sess.run(y, feed_dict=test_feed)
        result = sess.run(tf.arg_max(outputTenor, 1)+1)
        Id = sess.run(tf.to_int32(Id, name="ToInt32"))
        dataframe = pd.DataFrame({'Id': Id, 'Cover_Type': result}, columns=['Id', 'Cover_Type'])
        dataframe.to_csv("sample_submission.csv", index=False)

# if __name__ == '__main__':
#     tf.app.run()
main()




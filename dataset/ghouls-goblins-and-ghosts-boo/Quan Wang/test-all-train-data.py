#! /usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random


raw_train_data = pd.read_csv("../input/train.csv")
raw_test_data = pd.read_csv("../input/test.csv")


train_data = raw_train_data.copy()
del train_data["id"]
del train_data["color"]
Y_train = pd.get_dummies(raw_train_data[["type"]]).astype("float32").as_matrix()
X_train = train_data.drop("type", axis=1).astype("float32").as_matrix()



h, w = X_train.shape
x = tf.placeholder(tf.float32, shape=(None, w))
y_ = tf.placeholder(tf.float32, shape=(None, 3))
W = tf.Variable(tf.zeros([w, 3]))
b = tf.Variable(tf.zeros([3]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1500):
    batch_xs = X_train
    batch_ys = Y_train
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 10 == 0:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))


test_data = raw_test_data.copy()
del test_data["color"]
X_test = test_data.drop("id", axis=1)


test = tf.arg_max(y, 1) + 1
Y_test = sess.run(test, feed_dict={x: X_test})
test_data["type"] = Y_test
test_data['type'] = test_data['type'].astype(str).replace({'1': 'Ghost', '2': 'Goblin', '3': 'Ghoul'})
result = test_data.loc[:, ["id", "type"]]
print(result.head())

result.to_csv("result.csv", index=False)

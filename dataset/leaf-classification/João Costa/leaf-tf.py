import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import math
import random


INPUT_FOLDER = "../input"

with open(INPUT_FOLDER + "/train.csv", "r") as train:
    training_set = pd.read_csv(train)

training_set.fillna(0, inplace=True)
labels = pd.unique(training_set["species"])

with open(INPUT_FOLDER + "/test.csv", "r") as test:
    test_set = pd.read_csv(test)

test_set.fillna(0, inplace=True)


def extract_training_attributes(line):
    return (line[2:], line[1])


def extract_test_attributes(line):
    return line[1:]


# based on https://www.youtube.com/watch?v=sEciSlAClL8
def train_network(training_data, labels, runs, batch_size, epsilon, decay):
    num_attrs = len(training_data.columns) - 2
    num_labels = len(labels)
    num_hidden_nodes_l1 = int(num_attrs / 2)
    num_hidden_nodes_l2 = int(num_attrs / 4)

    X = tf.placeholder(tf.float32, [None, num_attrs])

    W1 = tf.Variable(tf.truncated_normal([num_attrs, num_hidden_nodes_l1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([num_hidden_nodes_l1]))

    #Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    Y1 = tf.nn.relu(tf.matmul(X, W1) + b1)

    W2 = tf.Variable(tf.truncated_normal([num_hidden_nodes_l1, num_hidden_nodes_l2], stddev=0.1))
    b2 = tf.Variable(tf.zeros([num_hidden_nodes_l2]))

    #Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + b2)
    Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)

    W3 = tf.Variable(tf.truncated_normal([num_hidden_nodes_l2, num_labels], stddev = 0.1))
    b3 = tf.Variable(tf.zeros([num_labels]))

    Y = tf.nn.softmax(tf.matmul(Y2, W3) + b3)
    Yt = tf.placeholder(tf.float32, [None, num_labels])

    global_step = tf.Variable(0, trainable=False)

    init = tf.global_variables_initializer()

    cross_entropy = -tf.reduce_sum(Yt * tf.log(tf.clip_by_value(Y, 0.00001, 1.0)))

    learning_rate = tf.train.exponential_decay(epsilon, global_step, runs, decay)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(cross_entropy, global_step=global_step)

    sess = tf.Session()
    sess.run(init)

    def create_y_vector(label):
        idx = labels.index(label)
        y = [0] * num_labels
        y[idx] = 1
        return y

    for i in range(runs):
        if i % 100 == 0:
            print("Running iteration " + str(i))
        train_batch = list(map(extract_training_attributes, training_data.sample(batch_size).values.tolist()))
        train_data = {X: list(map(lambda x: x[0], train_batch)), Yt: list(map(lambda x: create_y_vector(x[1]), train_batch))}
        sess.run(train_step, feed_dict=train_data)
        res_ce = sess.run(cross_entropy, feed_dict=train_data)
        if i % 100 == 0:
            print("Cross entropy:" + str(res_ce))
        if (res_ce < 0.05):
            break

    def classify(x):
        attrs = extract_test_attributes(x)
        pred = sess.run(Y, feed_dict={X: [attrs]})[0]
        res_dict = {}
        res_dict["id"] = int(x[0])
        for i in range(0, num_labels):
            res_dict[labels[i]] = pred[i]
        return res_dict
    return classify


classify = train_network(training_set, labels.tolist(), 30000, 250, 0.003, 0.90)
print("Finished training")
classifications = list(map(classify, test_set.values.tolist()))
print("Writing to file...")
pd.DataFrame(classifications).to_csv("predictions.csv", index=False)
print("Done")

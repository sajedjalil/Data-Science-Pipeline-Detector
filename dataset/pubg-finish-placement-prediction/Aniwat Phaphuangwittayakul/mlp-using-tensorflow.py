# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.model_selection import train_test_split
import os

version = "pubg"

df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

X_train = df_train.iloc[:, 3:25]
y_train = df_train.iloc[:, 25]

X_test = df_test.iloc[:, 3:]

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)

n_samples = X_train.shape[0]
n_features = X_train.shape[1] # equal to 22

scaler = MinMaxScaler(feature_range=(-1,1)).fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

y_train = np.reshape(y_train, [n_samples, 1])

print(X_train.shape, y_train.shape)

# Hyper parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 512
display_step = 1

# tf Graph Input
input_X = tf.placeholder(tf.float32, [None, n_features])
input_Y = tf.placeholder(tf.float32, [None, 1])

# Model architecture parametear
n_neurons_1 = 128
n_neurons_2 = 64
n_neurons_3 = 32
n_target = 1

def weight_initializer(shape):
  return tf.truncated_normal(shape, stddev=0.1)

def bias_initializer(shape):
  return tf.zeros(shape)

# Layer 1
W1 = tf.Variable(weight_initializer([n_features, n_neurons_1]))
b1 = tf.Variable(bias_initializer([n_neurons_1]))

# Layer 2
W2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
b2 = tf.Variable(bias_initializer([n_neurons_2]))

# Layer 3
W3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
b3 = tf.Variable(bias_initializer([n_neurons_3]))

# Layer 4
Wout = tf.Variable(weight_initializer([n_neurons_3, n_target]))
bout = tf.Variable(bias_initializer([n_target]))

# Hidden layer
keep_prob = tf.placeholder(tf.float32)

hidden_1 = tf.nn.relu(tf.add(tf.matmul(input_X, W1), b1))
dropout_1 = tf.nn.dropout(hidden_1, keep_prob)
hidden_2 = tf.nn.relu(tf.add(tf.matmul(dropout_1, W2), b2))
dropout_2 = tf.nn.dropout(hidden_2, keep_prob)
hidden_3 = tf.nn.relu(tf.add(tf.matmul(dropout_2, W3), b3))
dropout_3 = tf.nn.dropout(hidden_3, keep_prob)

# Make prediction
predictions = tf.nn.sigmoid(tf.add(tf.matmul(dropout_3, Wout), bout))

# Compute mean absolute error
cost = tf.reduce_mean(tf.abs(predictions - input_Y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    #save_path = saver.save(sess, "/tmp/model.ckpt")
    #ckpt = tf.train.latest_checkpoint('./model/' + version)
    #saver.restore(sess, save_path)

    for epoch in range(training_epochs):
        # Shuffle training data
        shuffle_indieces = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indieces]
        y_train = y_train[shuffle_indieces]
        total_batch = int(len(y_train) // batch_size)
        
        avg_cost = 0.
        
        # Minibatch training
        for i in range(0, total_batch):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]

            _, c = sess.run([optimizer, cost], feed_dict={input_X: batch_x, input_Y: batch_y, keep_prob: 0.9})
            # Compute average loss
            avg_cost += c / total_batch
        
        # Display loss
        if epoch % display_step == 0:
            #if not os.path.exists('./model/' + version):
            #  os.makedirs('./model/' + version)
            #saver.save(sess, './model/' + version + '/' + str(epoch))
            print("Epoch: {} Loss {}".format(epoch + 1, avg_cost))
            #pred = sess.run([predictions], feed_dict={input_X: X_train, keep_prob: 1.0})
            #print("Prediction value:",pred)

    
    print("Prediction state...")
    pred_test = sess.run([predictions], feed_dict={input_X: X_test, keep_prob: 1.0})
    # pred = pred.reshape(-1)
    pred_test = pred_test[0].ravel()
    pred_test = np.clip(pred_test, a_min=0, a_max=1)
    #print("Prediction for testing data is: ", pred_test)
    
    pred_test = pd.Series(pred_test, name='winPlacePerc')

    # pred = (pred + 1) / 2
    df_test['winPlacePerc'] = pred_test

    submission = df_test[['Id', 'winPlacePerc']]

    submission.to_csv('submission.csv', index=False)
    sess.close()

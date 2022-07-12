"""
Anomaly Detection Using Tensorflow

A first attempt at using Python for a kernel.
(Comments on Python good practices that are violated here are welcomed...)

Here we use an anomaly detection technique to see if the legit clicks (that are overwhelmingly
underrepresented) could be separated from the fraudulent ones. It has many similarities with
this kernel (in r): https://goo.gl/2acCNs

In more details, we rely on an autoencoder to reconstruct its input with the hope that a
pattern present in, and specific to, non-anomalistic data will be captured so that anomalistic
data (here the legit clicks) won't be reconstructed properly.

Our implementation reposes on Tensorflow.
"""

import tensorflow as tf
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

is_on_kaggle = True

if is_on_kaggle:
    input_path = '../input/'
else:
    input_path = './input/'

train = pd.read_csv(input_path + 'train.csv', nrows=50000000 if is_on_kaggle else 200000)
test = pd.read_csv(input_path + 'test.csv', nrows=None if is_on_kaggle else 1000000)

if is_on_kaggle:
    np.random.seed(1)
    train = train.sample(n=20000000)
    gc.collect()

response = pd.DataFrame()
response['is_attributed'] = train['is_attributed']
train.drop('is_attributed', axis=1, inplace=True)

train.drop(['attributed_time'], axis=1, inplace=True)

test_prediction = pd.DataFrame()
test_prediction['click_id'] = test['click_id']
test.drop('click_id', axis=1, inplace=True)

# convert the time information column into two new columns: weekday and hour
def timeEncoding(df):
    df['click_time'] = pd.to_datetime(df['click_time'])

    df['hour'] = df['click_time'].dt.hour
    df['weekday'] = df['click_time'].dt.weekday

    df.drop('click_time', axis=1, inplace=True)
    return df


train = timeEncoding(train)
test = timeEncoding(test)

# Frequency encoding of the ip
# (A bit dirty here... Besides, we use the test set, which obviously constitutes a malpractice.)
train_rows = np.shape(train)[0]
train = train.append(pd.DataFrame(data = test))
train['ip'] = train.ip.map(train.groupby('ip').size() / len(train))
test = train[train_rows:np.shape(train)[0]]
train = train[0:train_rows]
del train_rows

# normalize
scaler = StandardScaler()
scaler.fit(train)
train_dm = scaler.transform(train)

# free memory
del train
gc.collect()

# train only with non-anomaly
train_dm_non_anomalystic = train_dm[np.where(response == 0)[0]]
train_dm_anomalystic = train_dm[np.where(response == 1)[0]]

train_dm_non_anomalystic, validation_dm_non_anomalystic = train_test_split(train_dm_non_anomalystic, test_size=0.2)

# free memory
del train_dm
gc.collect()

niter =600 if is_on_kaggle else 1500
batch_size = 500000 if np.shape(train_dm_non_anomalystic)[0] > 1000000 else int(np.shape(train_dm_non_anomalystic)[0] / 5)
learning_rate = 0.004

number_of_features = len(train_dm_non_anomalystic[0])
number_of_neurons_first_layer = 6 
number_of_neurons_second_layer = 3 

# model (diabolo network)
We1 = tf.Variable(tf.random_normal([number_of_features, number_of_neurons_first_layer], dtype=tf.float32))
be1 = tf.Variable(tf.zeros([number_of_neurons_first_layer]))

We2 = tf.Variable(tf.random_normal([number_of_neurons_first_layer, number_of_neurons_second_layer], dtype=tf.float32))
be2 = tf.Variable(tf.zeros([number_of_neurons_second_layer]))

Wd1 = tf.Variable(tf.random_normal([number_of_neurons_second_layer, number_of_neurons_first_layer], dtype=tf.float32))
bd1 = tf.Variable(tf.zeros([number_of_neurons_first_layer]))

Wd2 = tf.Variable(tf.random_normal([number_of_neurons_first_layer, number_of_features], dtype=tf.float32))
bd2 = tf.Variable(tf.zeros([number_of_features]))

X = tf.placeholder(dtype=tf.float32, shape=[None, number_of_features])

encoding = tf.nn.tanh(tf.matmul(X, We1) + be1)
encoding = tf.matmul(encoding, We2) + be2
decoding = tf.nn.tanh(tf.matmul(encoding, Wd1) + bd1)
decoded = tf.matmul(decoding, Wd2) + bd2

loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(X, decoded))))
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# training
def get_batch(data, i, size):
    return data[range(i*size, (i+1)*size)]


tf.set_random_seed(1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
loss_train = []
loss_valid = []
loss_anomaly = []
for i in range(niter):
    if batch_size > 0:
        for j in range(np.shape(train_dm_non_anomalystic)[0] // batch_size):
            batch_data = get_batch(train_dm_non_anomalystic, j, batch_size)
            sess.run(train_step, feed_dict={X: batch_data})
    else:
        sess.run(train_step, feed_dict={X: train_dm_non_anomalystic})
    lt = sess.run(loss, feed_dict={X: train_dm_non_anomalystic})
    lv = sess.run(loss, feed_dict={X: validation_dm_non_anomalystic})
    la = sess.run(loss, feed_dict={X: train_dm_anomalystic})
    loss_train.append(lt)
    loss_valid.append(lv)
    loss_anomaly.append(la)
    if i % 50 == 0  or i == niter-1:
        print('iteration {0}: loss train = {1:.4f}, loss valid = {2:.4f}, loss anomaly = {3:.4f}'.format(i, lt, lv, la))

# result
plt.figure()
plt.clf()
plt.cla()
plt.semilogy(loss_train)
plt.semilogy(loss_valid)
plt.semilogy(loss_anomaly)
plt.ylabel('loss')
plt.xlabel('niter')
plt.legend(['train (w/o anomaly)', 'valid (w/o anomaly)', 'anomalistic data'], loc='upper right')
plt.gcf().savefig('learning_curves.png') if is_on_kaggle else plt.show()

# sample here to improve visibility
np.random.seed(1)
sampled_train_dm_non_anomalystic = train_dm_non_anomalystic[np.random.choice(len(train_dm_non_anomalystic), 5 * np.shape(train_dm_anomalystic)[0], replace=False)]
non_anomalystic_reconstructed = sess.run(decoded, feed_dict={X: sampled_train_dm_non_anomalystic})
anomalystic_reconstructed = sess.run(decoded, feed_dict={X: train_dm_anomalystic})

non_anomalystic_reconstructed_loss = np.sqrt(np.mean(np.square(sampled_train_dm_non_anomalystic - non_anomalystic_reconstructed), axis=1))
anomalystic_reconstructed_loss = np.sqrt(np.mean(np.square(train_dm_anomalystic - anomalystic_reconstructed), axis=1))

not_anomaly_loss = np.mean(non_anomalystic_reconstructed_loss)
anomaly_loss = np.mean(anomalystic_reconstructed_loss)
print('fraudulent:', not_anomaly_loss)
print('legit', anomaly_loss)

plt.figure()
bins = np.linspace(0, np.min(list([1.5, np.max(anomalystic_reconstructed_loss)])), 100)
plt.clf()
plt.cla()
plt.hist(non_anomalystic_reconstructed_loss, bins, color='red', alpha=0.5, label='fraudulent')
plt.hist(anomalystic_reconstructed_loss, bins, color='blue', alpha=0.5, label='legit')
plt.legend(loc='upper right')
plt.axvline(x=not_anomaly_loss, color='red', linestyle='--')
plt.axvline(x=anomaly_loss, color='blue', linestyle='--')
plt.gcf().savefig('legit-vs-fraudulent.png') if is_on_kaggle else plt.show()

# free memory
del train_dm_non_anomalystic, train_dm_anomalystic
gc.collect()

test_dm = scaler.transform(test)

# free memory
del test
gc.collect()

test_reconstructed = sess.run(decoded, feed_dict={X: test_dm})
test_reconstructed_loss = np.sqrt(np.mean(np.square(test_reconstructed - test_dm), axis=1))

# free memory
del test_dm
gc.collect()

# we uniformaly distribute the loss from 0 and 1
def distribute_in_range(data, min, max):
    max_data = np.max(data)
    a = (max - min) / (max_data - np.min(data))
    b = max - a * max_data
    return a * data + b

test_prediction['is_attributed'] = distribute_in_range(test_reconstructed_loss, 0, 1)

test_prediction.to_csv('test_prediction.csv', index=False)


# Copyright 2017 Loic Merckel
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and
# limitations under the License.
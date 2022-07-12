import kagglegym
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp

col = ['technical_30']

env = kagglegym.make()
o = env.reset()

im = pp.Imputer(strategy='median')
o.train[col] = im.fit_transform(o.train[col])
sX = pp.StandardScaler()
o.train[col] = sX.fit_transform(o.train[col])
o.train['b'] = 1

y_min = o.train.y.min()
y_max = o.train.y.max()

idx = (o.train.y<y_max) & (o.train.y>y_min)

features = ['b']+col
n = len(features)

learning_rate = 0.01
training_epochs = 1000
cost_history = np.empty(shape=[1],dtype=float)

X = tf.placeholder(tf.float32,[None,n])
Y = tf.placeholder(tf.float32,[None,1])
W = tf.Variable(tf.zeros([n,1]))

init = tf.global_variables_initializer()

y_ = tf.matmul(X, W)

cost = tf.add(tf.reduce_mean(tf.square(y_ - Y)), tf.reduce_mean(tf.square(W)))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(init)


for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={X: o.train[features], Y: o.train[['y']].values})

while True:
    o.features[col] = im.transform(o.features[col])
    o.features[col] = sX.transform(o.features[col])
    o.features['b'] = 1
    
    o.target.y = sess.run(y_, feed_dict={X:o.features[features]})
    o.target.y = np.clip(o.target.y, y_min, y_max)
    
    o, reward, done, info = env.step(o.target)
    if done:
        print(info)
        break
    if o.features.timestamp[0] % 100 == 0:
        print(reward)
import numpy as np
import os
import tensorflow as tf
import csv
from glob import glob


#Machine learning specific

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train_path = '../input/train-jpg/'
test_path = '../input/test-jpg/'

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/sample_submission.csv')


print(train.head())

image_paths = sorted(glob('../input/train-jpg/*.jpg'))[0:1000]
image_names = list(map(lambda row: row.split("/")[-1][:-4], image_paths))
print(image_names[0:10])




#print(train.iloc[:,1])

# Get the unique class names in the training dataset. 

#train(train.groupby(['tags']))
#print(train['tags'].unique())

#print(train['tags'].value_counts())

#Xfull, yfull = train.drop('image_name',axis =1), train['tags']

#print(y_full)


#Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xfull, yfull, test_size=0.1)

#rf = RandomForestClassifier(100)
#rf.fit(Xtrain,ytrain)
#print('Training score:',rf.score(Xtrain,ytrain))
#print('Validation score:', rf.score(Xvalid,yvalid))



# classes = []
# for val in train.iloc[:,1]:
#     if val not in classes:
#         print(train.groupby(val).count)
#         classes.append(val)




# train_data = csv.reader(train_path, delimiter=',', skipinitialspace=True) 

# category = []
# for row in category:
#     if row[1] not in category:
#         Category.append(row[1])    

# print (category) 

# get number of files in the folder
#for filename in os.listdir(train_path):
#    print (filename)
    
#print(len([name for name in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, name))]))
#print(len([name for name in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, name))]))


# logic starts here 

# Parameters
# learning_rate = 0.001
# training_iters = 200000
# batch_size = 128
# display_step = 10

# # Network Parameters
# n_input = 784 # data input (img shape: should be 28*28)

# #Number of classes
# #===================
# #Cloudy, Partly cloudy + primary, Primary, Haze+Primary, Agriculture+Roads+Habitation, Primary+Selective logging, Agriculture+Primary+PartlyCloudy, 
# #Habitation+Partly Cloudy, Agriculture+Roads+Primary, Water+Primary, Shifting Cultivation+Primary, Roads+Primary

# n_classes = 12 # total classes (0-9 digits)
# dropout = 0.75 # Dropout, probability to keep units

# # tf Graph input
# x = tf.placeholder(tf.float32, [None, n_input])
# y = tf.placeholder(tf.float32, [None, n_classes])
# keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# # Create some wrappers for simplicity
# def conv2d(x, W, b, strides=1):
#     # Conv2D wrapper, with bias and relu activation
#     x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
#     x = tf.nn.bias_add(x, b)
#     return tf.nn.relu(x)


# def maxpool2d(x, k=2):
#     # MaxPool2D wrapper
#     return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
#                           padding='SAME')


# # Create model
# def conv_net(x, weights, biases, dropout):
#     # Reshape input picture
#     x = tf.reshape(x, shape=[-1, 28, 28, 1])

#     # Convolution Layer
#     conv1 = conv2d(x, weights['wc1'], biases['bc1'])
#     # Max Pooling (down-sampling)
#     conv1 = maxpool2d(conv1, k=2)

#     # Convolution Layer
#     conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
#     # Max Pooling (down-sampling)
#     conv2 = maxpool2d(conv2, k=2)

#     # Fully connected layer
#     # Reshape conv2 output to fit fully connected layer input
#     fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
#     fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
#     fc1 = tf.nn.relu(fc1)
#     # Apply Dropout
#     fc1 = tf.nn.dropout(fc1, dropout)

#     # Output, class prediction
#     out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
#     return out

# # Store layers weight & bias
# weights = {
#     # 5x5 conv, 1 input, 32 outputs
#     'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
#     # 5x5 conv, 32 inputs, 64 outputs
#     'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
#     # fully connected, 7*7*64 inputs, 1024 outputs
#     'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
#     # 1024 inputs, 10 outputs (class prediction)
#     'out': tf.Variable(tf.random_normal([1024, n_classes]))
# }

# biases = {
#     'bc1': tf.Variable(tf.random_normal([32])),
#     'bc2': tf.Variable(tf.random_normal([64])),
#     'bd1': tf.Variable(tf.random_normal([1024])),
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }

# # Construct model
# #Get input data


# pred = conv_net(x, weights, biases, keep_prob)

# # Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# # Evaluate model
# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# # Initializing the variables
# init = tf.global_variables_initializer()

# # Launch the graph
# with tf.Session() as sess:
#     sess.run(init)
#     step = 1
#     # Keep training until reach max iterations
#     while step * batch_size < training_iters:
#         batch_x, batch_y = mnist.train.next_batch(batch_size)
#         # Run optimization op (backprop)
#         sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
#                                       keep_prob: dropout})
#         if step % display_step == 0:
#             # Calculate batch loss and accuracy
#             loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
#                                                               y: batch_y,
#                                                               keep_prob: 1.})
#             print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
#                   "{:.6f}".format(loss) + ", Training Accuracy= " + \
#                   "{:.5f}".format(acc))
#         step += 1
#     print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
    #                                   y: mnist.test.labels[:256],
    #                                   keep_prob: 1.}))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import tensorflow as tf
from sklearn import preprocessing

num_examples = 15120
learning_rate = 0.03

n_STEP = 4000      #训练8000轮
step = 15120       #每次输入的数据个数
n_hidden_1 = 128   #第一层隐藏节点的个数
n_hidden_2 = 128   #第二层隐藏节点个个数
n_hidden_3 = 128
n_input = 54       #输入特征
n_classes =7       #7个类别
reg=0.01           #正则化参数
keep_prob = 0.8    #丢弃部分神经元

data = pd.read_csv("../input/train.csv")
x_train = data.iloc[:,1:-1]
y_train = data['Cover_Type']

x_train_batch =tf.convert_to_tensor(x_train)
y_train_batch =tf.convert_to_tensor(y_train)

x = tf.placeholder("float",[None,n_input])
y = tf.placeholder("float",[None,n_classes])

#创建模型
def multilayer_perceptron(x,weights,biases):
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1_drop = tf.nn.dropout(layer_1,keep_prob)
    layer_1_drop = tf.nn.relu(layer_1_drop)


    layer_2 = tf.add(tf.matmul(layer_1_drop,weights['h2']),biases['b2'])
    layer_2_drop = tf.nn.dropout(layer_2,keep_prob)
    layer_2_drop = tf.nn.relu(layer_2_drop)
    
    layer_3 = tf.add(tf.matmul(layer_2,weights['h3']),biases['b3'])
    layer_3_drop = tf.nn.dropout(layer_3,keep_prob)
    layer_3_drop = tf.nn.relu(layer_3_drop)


    out_layer = tf.add(tf.matmul(layer_3_drop,weights['out']),biases['out'])
    return out_layer
#学习参数
weights = {'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
           'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
           'h3':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
           'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))}
biases =  {'b1':tf.Variable(tf.random_normal([n_hidden_1])),
          'b2':tf.Variable(tf.random_normal([n_hidden_2])),
          'b3':tf.Variable(tf.random_normal([n_hidden_3])),        
          'out':tf.Variable(tf.random_normal([n_classes]))}
pred = multilayer_perceptron(x,weights,biases)



#定义损失函数和优化器
#计算输出与实际标签的交叉熵，计算损失对向量求均值
global_step = tf.Variable(0,trainable=False)
decaylearning_rate = tf.train.exponential_decay(learning_rate,global_step,200,0.95)
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y)) + tf.nn.l2_loss(weights['h1'])*reg + tf.nn.l2_loss(weights['h2'])*reg + tf.nn.l2_loss(weights['h3'])*reg 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(decaylearning_rate).minimize(cost,global_step=global_step)
#每次读入500个数据

init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()
lb = preprocessing.LabelBinarizer()

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(local_init_op)
    # Start populating the filename queue.
        # while not coord.should_stop():
    for i in range(n_STEP):
            example, label = sess.run([x_train_batch, y_train_batch])
            # print(tf.shape(example))
            # print(label)
            c = tf.ones(shape=(step,))
            label = tf.to_float(label) 
            #生成onehot编码,由于标签从1开始，所以要先减1
            label = tf.subtract(label,c)
            batch_size =tf.size(label)
            #expand_dims方法为在指定位置插入1
            labels = tf.expand_dims(label, 1)
            #转化为整形
            labels = tf.to_int32(labels)
            # print(sess.run(labels))
            indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
            # print(sess.run(indices))
            concated = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, n_classes]), 1.0, 0.0)
            lab_y=sess.run(onehot_labels)
            # print(tf.shape(onehot_labels))
            sess.run([optimizer, cost], feed_dict={x: example, y: lab_y})
            if i%200==0:
                print(i)
                print(sess.run([decaylearning_rate]))
                print(cost.eval({x: example, y: lab_y}))
    data_test = pd.read_csv('../input/test.csv')
    x_test = data_test.iloc[:,1:]
    x_test_batch =tf.convert_to_tensor(x_test)
    test_examples = sess.run(x_test_batch)
    sess.run(pred, feed_dict={x: test_examples})
    preds = tf.argmax(pred,1)
    preds = preds.eval({x: test_examples}) + 1
    print(preds)
sub = pd.DataFrame({"Id": data_test.iloc[:,0].values,"Cover_Type": preds})
sub.to_csv("sample_submission.csv", index=False) 
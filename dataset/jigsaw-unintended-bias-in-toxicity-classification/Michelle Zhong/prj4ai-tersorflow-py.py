from utils import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
import pandas as pd
import numpy as np
from collections import *
from sklearn.metrics import *
from sklearn.datasets import *
import random
from nltk.corpus import stopwords
import re

# Preprocessing & utility functions
english_stopwords = stopwords.words('english')
def clearstring(string):
    string = re.sub('[^A-Za-z0-9 ]+', '', string)
    string = string.split(' ')
    string = filter(None, string)
    string = [y.strip() for y in string if y.strip() not in english_stopwords]
    string = ' '.join(string)
    return string.lower()

def separate_dataset(trainset, ratio = 0.5):
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split('\n')
        data_ = list(filter(None, data_))
        data_ = random.sample(data_, int(len(data_) * ratio))
        for n in range(len(data_)):
            data_[n] = clearstring(data_[n])
        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget
    
# load the filted 2 text files, which are put under the path
trainset = load_files(container_path = '../input/alldata4ai/data_all/data_all/', encoding = 'UTF-8')
trainset.data, trainset.target = separate_dataset(trainset,1.0)

# check target name to classify
#print (trainset.target_names)
# check whether len of target is equal to the len of data
#print (len(trainset.data))
#print (len(trainset.target))

# build dictionary for the text in order to transform the words to vectors
def build_dataset(words, n_words):
    # count -> most frequently used tokens
    count = [['GO', 0], ['PAD', 1], ['EOS', 2], ['UNK', 3]]
    count.extend(Counter(words).most_common(n_words - 1))
    # word -> int
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    # data -> corpus
    data = list()
    # unk refers to unknown tokens
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    # int -> word
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

# split the dataset
ONEHOT = np.zeros((len(trainset.data),len(trainset.target_names)))
ONEHOT[np.arange(len(trainset.data)),trainset.target] = 1.0
train_X, test_X, train_Y, test_Y, train_onehot, test_onehot = train_test_split(trainset.data, 
                                                                               trainset.target, 
                                                                               ONEHOT, test_size = 0.2)
                                                                               
# get all tokens in the text
concat = " ".join(trainset.data).split()
vocabulary_size = len(list(set(concat)))
# get the dictionary
data, count, dictionary, rev_dictionary = build_dataset(concat,vocabulary_size)
print('vocab from size: %d'%(vocabulary_size))
print('Most common words (+UNK)', count[1:10])
#print('Sample data', data[:10])

# release the memory
import gc
del concat
gc.collect()

# emmmm...release it again in case lack of memory
gc.collect()

# define model
class Model:
    def __init__(self, size_layer, num_layers, embedded_size,
                 dict_size, dimension_output, learning_rate, attention_size=150):
        
        # 128 units in a LSTM cell
        def cells(reuse=False):
            return tf.nn.rnn_cell.LSTMCell(size_layer,initializer=tf.orthogonal_initializer(),reuse=reuse)
        # for input data, appear with sess.run: feed_dict{} 
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        # embedding layer by dicitonary
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        # add emedding layer
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        # LSTM + RNN
        rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
        # creates a recurrent neural network specified by RNNCell cell
        outputs, last_state = tf.nn.dynamic_rnn(rnn_cells, encoder_embedded, dtype = tf.float32)
        # attentino layer
        attention_w = tf.get_variable("attention_v", [attention_size], tf.float32)
        # inserts a dimension of 1 into a tensor's shape
        # interface for the densely-connected layer
        query = tf.layers.dense(tf.expand_dims(last_state[-1].h, 1), attention_size)
        keys = tf.layers.dense(outputs, attention_size)
        align = tf.reduce_sum(attention_w * tf.tanh(keys + query), [2])
        # softmax layer
        align = tf.nn.softmax(align)
        # output layer
        outputs = tf.squeeze(tf.matmul(tf.transpose(outputs, [0, 2, 1]),
                                             tf.expand_dims(align, 2)), 2)
        W = tf.get_variable('w',shape=(size_layer, dimension_output),initializer=tf.orthogonal_initializer())
        b = tf.get_variable('b',shape=(dimension_output),initializer=tf.zeros_initializer())
        self.logits = tf.matmul(outputs, W) + b
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.Y))
        # adam optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
size_layer = 128
num_layers = 4
embedded_size = 128
dimension_output = 2 # binary classification
learning_rate = 1e-3 
maxlen = 250 # max number of words in each comment
batch_size = 256

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(size_layer,num_layers,embedded_size,vocabulary_size+4,dimension_output,learning_rate)
sess.run(tf.global_variables_initializer())

# todo: remove explicit for loop
def str_idx(corpus, dic, maxlen, UNK=3):
    X = np.zeros((len(corpus),maxlen))
    rowN = 0
    for row in corpus:
        colN = 0
        for k in row.split(): 
            if colN < maxlen:
                val = dic[k] if k in dic else UNK
                X[rowN, colN]= val
                colN += 1
            else:
                break
        rowN += 1
    return X

# train the model
EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 3, 0, 0, 0
while True:
    lasttime = time.time()
    # reach the epoch times for early stopping in case of overfitting
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print('break epoch:%d\n'%(EPOCH))
        break
        
    # train the model for each batch
    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    for i in range(1, (len(train_X) // batch_size) * batch_size, batch_size):
        # get the batch_x represented by bow model
        # 256 X 250
        batch_x = str_idx(train_X[i:i+batch_size],dictionary,maxlen)
        acc, loss, _ = sess.run([model.accuracy, model.cost, model.optimizer], 
                           feed_dict = {model.X : batch_x, model.Y : train_onehot[i:i+batch_size]})
        train_loss += loss
        train_acc += acc
    
    # test the model
    for i in range(0, (len(test_X) // batch_size) * batch_size, batch_size):
        # encode the batch_x by bow model
        batch_x = str_idx(test_X[i:i+batch_size],dictionary,maxlen)
        acc, loss = sess.run([model.accuracy, model.cost], 
                           feed_dict = {model.X : batch_x, model.Y : test_onehot[i:i+batch_size]})
        test_loss += loss
        test_acc += acc
    
    # update the loss and accuarcy
    train_loss /= (len(train_X) // batch_size)
    train_acc /= (len(train_X) // batch_size)
    test_loss /= (len(test_X) // batch_size)
    test_acc /= (len(test_X) // batch_size)
    
    # if acc has been improved, set the checkpoint to 0
    if test_acc > CURRENT_ACC:
        print('epoch: %d, pass acc: %f, current acc: %f'%(EPOCH,CURRENT_ACC, test_acc))
        CURRENT_ACC = test_acc
        CURRENT_CHECKPOINT = 0
    # no improvement, add the record
    else:
        CURRENT_CHECKPOINT += 1
    
    # output info   
    print('time taken:', time.time()-lasttime)
    print('epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n'%(EPOCH,train_loss,
                                                                                          train_acc,test_loss,
                                                                                          test_acc))
    EPOCH += 1
 
# cannot run due to out of memory   
#logits = sess.run(model.logits, feed_dict={model.X:str_idx(test_X,dictionary,maxlen)})
#print(classification_report(test_Y, np.argmax(logits,1), target_names = trainset.target_names))

# todo lists:
# 1. Introduce embedding layers by glove and other pre-trained models
# 2. Remove explicit for-loop
# 3. Do more EDA before utilizing the model
# 4. Add more columns in the original dataset like identity (e.g. "hetersexual, Chirstian") info to improve acc
# 5. Further tuning
# 6. Try more nn stuctures
# 7. Output the specific values rather than doing binary classification
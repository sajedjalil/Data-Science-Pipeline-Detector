# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.losses import binary_crossentropy
from collections import Counter



# Any results you write to the current directory are saved as output.
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAXLEN = 1000
EMDED_SIZE = 50
MODEL_DEPTH = 6
UNKNOWN_CHAR = 'ⓤ'
PAD_CHAR = '℗'
BSIZE = 512
EPOCHS = 4

train_data = pd.read_csv("../input/train.csv")
#train_data = train_data.sample(frac = 0.1)
test_data = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")

sentences_train = train_data["comment_text"].fillna("_NAN_").values
sentences_test = test_data["comment_text"].fillna("_NAN_").values


def create_char_vocabulary(texts,min_count_chars=100):
    counter = Counter()
    for k, text in enumerate(texts):
        counter.update(text)

    raw_counts = list(counter.items())
    print(raw_counts)

    print('%s characters found' %len(counter))
    print('keepin characters with count > %s' % min_count_chars)
    vocab = [char_tuple[0] for char_tuple in raw_counts if char_tuple[1] > min_count_chars]
    char2index = {char:(ind+1) for ind, char in enumerate(vocab)}
    char2index[UNKNOWN_CHAR] = 0
    char2index[PAD_CHAR] = -1
    index2char = {ind:char for char, ind in char2index.items()}
    print('%s remaining characters' % len(char2index))
    return char2index, index2char
    
def char2seq(texts, maxlen):
    res = np.zeros((len(texts),maxlen))
    for k,text in enumerate(texts):
        seq = np.zeros((len(text))) #equals padding with PAD_CHAR
        for l, char in enumerate(text):
            try:
                id = char2index[char]
                seq[l] = id
            except KeyError:
                seq[l] = char2index[UNKNOWN_CHAR]
        seq = seq[:maxlen]
        res[k][:len(seq)] = seq
    return res

char2index, index2char = create_char_vocabulary(sentences_train)

import json
with open('index.json', 'w', encoding='utf-8') as f:
    json.dump(char2index, f, ensure_ascii=False, indent=4)

X_train = char2seq(sentences_train,MAXLEN)
X_test = char2seq(sentences_test,MAXLEN)
Y_train = train_data[list_classes].values

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(dtype=tf.int32,shape=(None,MAXLEN))
    y = tf.placeholder(dtype=tf.float32,shape=(None,6))
    is_training = tf.placeholder(tf.bool, [], name='is_training')

    embedding = tf.get_variable("embedding", [len(char2index), EMDED_SIZE], dtype=tf.float32)
    x2 = tf.nn.embedding_lookup(embedding, x, name="embedded_input")

    for i in range(3,3+MODEL_DEPTH):
        x2 = tf.layers.conv1d(x2, filters=2**i, kernel_size=3, strides=1)
        x2 = tf.layers.conv1d(x2, filters=2**i, kernel_size=3, strides=1)
        x2 = tf.layers.max_pooling1d(x2, pool_size=2, strides=2)

    x2 = tf.reduce_mean(x2, axis=1)
    x2 = tf.contrib.layers.fully_connected(x2, 64, activation_fn=tf.nn.relu)
    logits = tf.contrib.layers.fully_connected(x2, 6, activation_fn=tf.nn.sigmoid)

    loss = tf.losses.log_loss(labels=y,predictions=logits)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    
    (_, auc_update_op) = tf.metrics.auc(labels=y,predictions=logits,curve='ROC')    
    
train_iters = len(X_train) - BSIZE
with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(EPOCHS+1):
        step = 0
        tf.local_variables_initializer().run(session=sess)
        while step * BSIZE < train_iters:
            batch_x = X_train[step * BSIZE:(step + 1) * BSIZE]
            batch_y = Y_train[step * BSIZE:(step + 1) * BSIZE]
            logloss , _, roc_auc = sess.run([loss,optimizer,auc_update_op],feed_dict={x:batch_x,
                                                             y:batch_y,
                                                             is_training:True})

            print('e%s -- s%s -- logloss: %s -- roc_auc: %s' %(epoch,step,logloss,roc_auc))
            step +=1
            
    num_batches = (len(X_test) // BSIZE) + 1
    res = np.zeros((len(X_test), 6))
    for s in range(num_batches):
        if s % 50 == 0:
            print(s)
        batch_x_test = X_test[s * BSIZE:(s + 1) * BSIZE]
        logits_ = sess.run(logits, feed_dict={x: batch_x_test,
                                              is_training:False})

        res[s * BSIZE:(s + 1) * BSIZE] = logits_
    
    sample_submission[list_classes] = res

    fn = 'submission.csv'
    sample_submission.to_csv(fn, index=False)
    print(sess.run(logits, feed_dict = {x: char2seq(["fuck you"], MAXLEN), 
                                        is_training:False}))
    
    print(sess.run(logits, feed_dict = {x: char2seq(["COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK"], MAXLEN), 
                                        is_training:False}))
    
    converter = tf.lite.TFLiteConverter.from_session(sess, input_tensors=[x], output_tensors=[logits])
    tflite_model = converter.convert()
    open("converted_model_5_epoch.tflite", "wb").write(tflite_model)
    
        
#     x = tf.placeholder(dtype=tf.int32,shape=(None,MAXLEN))
#     y = tf.placeholder(dtype=tf.float32,shape=(None,6))
        
# with tf.Session(graph=graph) as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     logits_ = sess.run(logits, feed_dict = {x: char2seq(["COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK"], MAXLEN), 
#                                         is_training:False})    
#     converter = tf.lite.TFLiteConverter.from_session(sess, input_tensors=[x], output_tensors=[logits])
#     tflite_model = converter.convert()
#     open("converted_model_5_epoch.tflite", "wb").write(tflite_model)
    
# with tf.Session(graph=graph) as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     print(sess.run(logits, feed_dict = {x: char2seq(["fuck you"], MAXLEN), 
#                                         is_training:False}))
    
# import json
# with open('index.json', 'w', encoding='utf-8') as f:
#     json.dump(char2index, f, ensure_ascii=False, indent=4)
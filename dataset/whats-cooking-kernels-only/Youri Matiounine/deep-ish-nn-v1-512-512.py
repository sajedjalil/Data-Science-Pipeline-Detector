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

import time
import gc
import re
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer


# read raw data
start_time = time.time()
input_dir = os.path.join(os.pardir, 'input')
print('  Loading data...')
train_df = pd.read_json('../input/train.json') # store as dataframe objects
test_df = pd.read_json('../input/test.json')
print('    Time elapsed %.0f sec'%(time.time()-start_time))


# apply label encoding on the target variable
len_train = train_df.shape[0]
target = train_df['cuisine']
merged_df = pd.concat([train_df, test_df], sort=True)
lb = LabelEncoder()
target2 = lb.fit_transform(target)


# preprocess labels
print('  Pre-processing data...')
features_processed = []
for item in merged_df['ingredients']:
    newitem = []
    for ingr in item:
        ingr = ingr.lower() # to lowercase 
        ingr = re.sub("[^a-zA-Z]"," ",ingr) # remove punctuation, digits or special characters 
        ingr = re.sub((r'\b(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\b'), ' ', ingr) # remove different units
        ingr = re.sub((r'\b(style|light|swanson|vegetarian|progresso|homemade|hellmann|hidden valley|johnsonville|wish bone|lipton|refrigerated|vegan|salted|original|mccormick|unsweetened|taco bell|medium|small|old el paso|unsalted)\b'), ' ', ingr) # remove some brand names
        ingr = re.sub("  "," ",ingr) # remove double space
        newitem.append(ingr)
    features_processed.append(newitem)
merged_df['ingredients'] = features_processed # put it back
print('    Time elapsed %.0f sec'%(time.time()-start_time))


# text data features
print ("prepare text data ... ")
d = merged_df['ingredients'][:len_train]
train_text = [" ".join(doc) for doc in d]
d = merged_df['ingredients'][len_train:]
test_text = [" ".join(doc) for doc in d]
del d
gc.collect()


# Feature Engineering
print ("TF-IDF on text data ... ")
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(binary=True)
X = tfidf.fit_transform(train_text).astype(np.float32)
X_test = tfidf.transform(test_text).astype(np.float32)
del train_text, test_text
gc.collect()


# use NN
import tensorflow as tf
from random import randint

L2c = 1e-3              # loss, with L2: output = sum(W ** 2)/2
lr0 = 0.02              # starting learning rate
lr_decay = 0.985        # lr decay rate
iterations = 200        # full passes over data
L1 = 512                # level 1 neurons
L2 = 512                # level 2 neurons
NUMV = X.shape[1]

# loop over cross-validation folds
oof_preds = np.zeros([X.shape[0]])
sub_preds = np.zeros([X_test.shape[0],20])
X = X.todense()
target2 = target2.astype(np.int32)
num_folds = 5
folds = KFold(n_splits=num_folds, shuffle=True, random_state=4564)
xt = X_test.todense()
yt = np.zeros([xt.shape[0]])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X)):
    # define inputs
    y_ = tf.placeholder(tf.int32, [None])         # result
    x  = tf.placeholder(tf.float32, [None, NUMV]) # x

    # init coeffs: values
    W0v = np.sqrt(1/1000) * np.random.randn(NUMV, L1)
    W0bv= 0.001 * np.ones([1, L1])
    W1v = np.sqrt(1/1000) * np.random.randn(L1, L2)
    W1bv= 0.001 * np.ones([1, L2])
    W2v = np.sqrt(1/1000) * np.random.randn(L2, 20)

    # init coeffs: variables
    W0  = tf.Variable(W0v.astype(np.float32))
    W0b = tf.Variable(W0bv.astype(np.float32))
    W1  = tf.Variable(W1v.astype(np.float32))
    W1b = tf.Variable(W1bv.astype(np.float32))
    W2  = tf.Variable(W2v.astype(np.float32))

    # model - 2 hidden layers
    x1 = tf.nn.relu( tf.matmul( x, W0 ) + W0b ) # None x L1
    x2 = tf.nn.relu( tf.matmul( x1, W1 ) + W1b )# None x L2
    y  = tf.matmul( x2, W2 )                    # None x 20
    y_pred = tf.argmax( y, axis=1 )             # None

    # training
    loss1 = L2c * ( tf.nn.l2_loss( W0 ) + tf.nn.l2_loss( W1 ) )
    loss0 = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( labels=y_, logits=y ) )
    loss  = loss0 + loss1
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(lr0, global_step, 1, lr_decay) 
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    x0 = X[train_idx,:]
    y0 = target2[train_idx]
    fd0 = {y_: y0, x: x0}
    x1 = X[valid_idx,:]
    y1 = target2[valid_idx]
    fd1 = {y_: y1, x: x1}
    print('start training loop...')
    for i in range(iterations):
        _,l,l1,lrv,yp = sess.run( [train_step, loss0, loss1, learning_rate, y_pred], feed_dict=fd0 )
        if i%20 == 0:
            l2,yp2 = sess.run( [loss0, y_pred], feed_dict=fd1 )
            a = (yp==y0).mean()
            a2 = (yp2==y1).mean()
            print('iter,%d, loss,%.4f,%.4f,%.4f, time,%.0f sec, lr,%.4f, accuracy:%.3f/%.3f'%(i,l,l1,l2,time.time()-start_time,lrv,a,a2))
    oof_preds[valid_idx] = yp2
    fd2 = {y_: yt, x: xt}
    pr2 = sess.run( y, feed_dict=fd2 )
    sub_preds += pr2 / folds.n_splits
    sess.close()
sub_preds = sub_preds.argmax(axis=1)
e = (target2==oof_preds).mean()
print('Full validation score/error %.4f/%.4f' %(e,1-e))
print('    Time elapsed %.0f sec'%(time.time()-start_time))

# Write submission file
out_df = pd.DataFrame({'id': merged_df['id'][len_train:]})
pred2 = lb.inverse_transform(sub_preds)
out_df['cuisine'] = pred2
out_df.to_csv('submission.csv', index=False)
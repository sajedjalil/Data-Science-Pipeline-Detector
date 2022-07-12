## TensorFlow NN for Santander dataset
##
## WARNING: unstable, buggy, and probably scores poorly on leaderboard
##
## Adapted from: https://www.kaggle.com/kakauandme/digit-recognizer/tensorflow-deep-nn/
## Two functions, "tied_rank" and "auc", were copied verbatim from:
## https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py

import numpy as np
import pandas as pd

#%matplotlib inline                 # uncomment if converting to iPython notebook 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
import time
start_time = time.time()
tag = str(start_time)
np.random.seed(200)
# settings
DEBUG = False
GRAPH = False
EPSILON = 1e-12
NUM_NEG_EXAMPLES = 73012 # number of negative examples to use; must not exceed 73012
                        # (the number of positive examples in the training data is 3008)
LEARNING_RATE = 5e-4
TRAINING_ITERATIONS = 6000
    
HIDDEN_LAYER_SIZE = [392,169] #We use two hidden layers
DROPOUT = 0.5
BATCH_SIZE = 75

# set to 0 to train on all available data
VALIDATION_SIZE = 15000

# read training data from CSV file 
data = pd.read_csv('../input/train.csv')
train_ids_and_targets = data[['ID','TARGET']]

pos = data.loc[data['TARGET'] == 1] # "positive" examples, i.e., those with TARGET=1
neg = data.loc[data['TARGET'] == 0] # "negative" training examples...
if DEBUG:
    print('pos:')
    print(pos.head())
    print('neg:')
    print(neg.head())

neg_rows = neg.values[:,1:-1] #discard ID and TARGET
pos_rows = pos.values[:,1:-1] #discard ID and TARGET
neg_labels = neg.values[:,[0,-1]] #ID and TARGET only
pos_labels = pos.values[:,[0,-1]] #ID and TARGET only

if DEBUG:
    print('neg_rows:',neg_rows.shape)
    print('pos_rows:',pos_rows.shape)

# shuffle the data
num_examples = neg.shape[0]
perm = np.arange(num_examples) 
np.random.shuffle(perm) #a random permutation
neg_rows = neg_rows[perm]       #apply the same permutation to the feature rows
neg_labels = neg_labels[perm]   #and the TARGET values
if DEBUG:
    print('after shuffling:')
    print('pos:')
    print(pos_rows[:5])
    print('neg:')
    print(neg_rows[:5])
    print(pos_rows.shape,pos_labels.shape)
    print(neg_rows.shape,neg_labels.shape)

# restrict training data by discarding all but NUM_NEG_EXAMPLES of the negative examples
# but keep all 3008 positive examples
data = np.concatenate((pos_rows,neg_rows[:NUM_NEG_EXAMPLES,:]),axis=0)
labels = np.concatenate((pos_labels,neg_labels[:NUM_NEG_EXAMPLES,:]),axis=0)

# calculate the quartiles on the training data for normalization
mu = np.percentile(data,50.0,axis=0)
s2 = np.percentile(data,75.0,axis=0)
s1 = np.percentile(data,25.0,axis=0)

if DEBUG:
    print('data, labels:',data.shape,labels.shape)
    print('median:',mu.shape)
    print('quartiles:',s1.shape,s2.shape)

# normalize training data
if DEBUG:
    print('normalizing training data...')
data_norm = (data - mu) / (1 + s2 - s1)
dataRows = data_norm.astype(np.float)
row_length = len(dataRows[0])

# read testing data from CSV file 
if DEBUG:
    print('reading testing data...')
test = pd.read_csv('../input/test.csv')
test_ids = test[['ID']]                     #store the ID column
test.drop('ID', axis=1, inplace=True)       #discard the ID column
test = test.values
# normalize testing data
if DEBUG:
    print('normalizing testing data...')
test_rows = (test - mu) / (1 + s2 - s1)
test_rows = test_rows.astype(np.float)

if DEBUG:
    print('train:')
    print(dataRows[:5])
    print('test:')
    print(test_rows[:5])
    print('test_rows({0[0]},{0[1]})'.format(test_rows.shape))
    print('row length',row_length)

labels_flat = labels[:,1]
if DEBUG:
    print('labels_flat({0})'.format(len(labels_flat)))

labels_count = np.unique(labels_flat).shape[0]
if DEBUG:
    print('labels_count => {0}'.format(labels_count))

# For most classification problems "one-hot vectors" are used. A one-hot vector 
# is a vector that contains a single element equal to 1 and the rest of the 
# elements equal to 0. In this case, the *nth* digit is represented as a zero 
# vector with 1 in the *nth* position.
# convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]
#
# We only have a binary classification problem, but we still use "one-hot vectors"
# of length 2.
# 0 => [1 0]
# 1 => [0 1]

# this function is no longer called
def dense_to_one_hot(labels_dense, num_classes):

    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    i = [x+y for (x,y) in zip(index_offset,labels_dense.ravel())]
    labels_one_hot.flat[i] = 1
    # for some reason I needed to explicitly create the iterator i
    # the syntax used in the original script was causing an error
    # IndexError: unsupported iterator index
    # it seems this may be a bug in numpy, but I haven't verified this
    if DEBUG:
        print('labels_one_hot',labels_one_hot.shape)
        print('labels_one_hot.flat',labels_one_hot.flat[:].shape)
        print('----')
        print('index_offset',index_offset.shape)
        print('index_offset[:5]',index_offset[:5])
        print('labels_dense.ravel()',labels_dense.ravel().shape)
        print('i[:10]',i[:10])
        print('labels_dense[:5]',labels_dense[:5])
        print('labels_one_hot[:5,:]',labels_one_hot[:5,:])
    return labels_one_hot

def tied_rank(x):
    """
    Computes the tied rank of elements in x.
    This function computes the tied rank of elements in x.
    Parameters
    ----------
    x : list of numbers, numpy array
    Returns
    -------
    score : list of numbers
            The tied rank f each element in x
    """
    x_p = [(u,i) for i,u in enumerate(x)]
    sorted_x = sorted(x_p)
    r = [0 for k in x]
    cur_val = sorted_x[0][0]
    last_rank = 0
    for i in range(len(sorted_x)):
        if cur_val != sorted_x[i][0]:
            cur_val = sorted_x[i][0]
            for j in range(last_rank, i): 
                r[sorted_x[j][1]] = float(last_rank+1+i)/2.0
            last_rank = i
        if i==len(sorted_x)-1:
            for j in range(last_rank, i+1): 
                r[sorted_x[j][1]] = float(last_rank+i+2)/2.0
    return r

def auc(actual, posterior):
    """
    Computes the area under the receiver-operater characteristic (AUC)
    This function computes the AUC error metric for binary classification.
    Parameters
    ----------
    actual : list of binary numbers, numpy array
             The ground truth value
    posterior : same type as actual
                Defines a ranking on the binary numbers, from most likely to
                be positive to least likely to be positive.
    Returns
    -------
    score : double
            The mean squared error between actual and posterior
    """
    r = tied_rank(posterior)
    num_positive = len([0 for x in actual if x==1])
    num_negative = len(actual)-num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if actual[i]==1])
    auc = ((sum_positive - num_positive*(num_positive+1)/2.0) /
           (num_negative*num_positive))
    return auc



labels = labels_flat #dense_to_one_hot(labels_flat, labels_count)
labels_count = 1 # THIS IS A HACK
labels = labels.astype(np.uint8)
if DEBUG:
    #print('labels({0[0]},{0[1]})'.format(labels.shape))
    print('labels:',labels.shape)
    print(np.sum(labels,axis=0))


# Lastly we set aside data for validation. It's essential in machine
# learning to have a separate dataset which doesn't take part in the
# training and is used to make sure that what we've learned can actually be generalised.

# permute again
n_rows = dataRows.shape[0]
perm   = np.arange(n_rows)
np.random.shuffle(perm)
np.random.shuffle(perm)
dataRows = dataRows[perm]
labels   = labels[perm]

# split data into training & validation
# (we don't allow using more than half the data for validation)
k = min(n_rows//2,VALIDATION_SIZE)
validation_rows = dataRows[:k]
validation_labels = labels[:k]

train_rows = dataRows[k:]
train_labels = labels[k:]


# *Data is ready. The neural network structure is next.*
# ## TensorFlow graph
# TensorFlow does its heavy lifting outside Python. Therefore, instead of 
# running every single operation independently, TensorFlow allows users to 
# build a whole graph of interacting operations and then runs the workflow 
# in a separate process at once.
#
# #### Helper functions
# For this NN model, a lot of weights and biases are created. Generally, 
# weights should be initialised with a small amount of noise for symmetry 
# breaking, and to prevent 0 gradients. 
# 
# Since we are using [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks) 
# neurons (ones that contain rectifier function *f(x)=max(0,x)*), it is 
# also good practice to initialise them with a slightly positive initial 
# bias to avoid "dead neurons".

# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.abs(tf.truncated_normal(shape, stddev=0.01))#tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# input & output of NN

# input rows
x = tf.placeholder('float', shape=[None, row_length])
# labels (output)
y_ = tf.placeholder('float', shape=[None,labels_count])

# densely connected layer
W_fc1 = weight_variable([row_length, HIDDEN_LAYER_SIZE[0]])
b_fc1 = bias_variable([1,HIDDEN_LAYER_SIZE[0]])


x_flat = x #tf.reshape(x, [None, row_length])

x_fc1 = tf.sigmoid(tf.matmul(x_flat, W_fc1) + b_fc1) #tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)

# To prevent overfitting, we  apply 
# [dropout](https://en.wikipedia.org/wiki/Convolutional_neural_network#Dropout) 
# before the readout layer.
# 
# Dropout removes some nodes from the network at each training stage. 
# Each of the nodes is either kept in the network with probability *keep_prob* 
# or dropped with probability *1 - keep_prob*. After the training stage is over 
# the nodes are returned to the NN with their original weights.
#
# dropout
keep_prob = tf.placeholder('float')
x_fc1_drop = tf.nn.dropout(x_fc1, keep_prob)

W_fc2 = weight_variable([HIDDEN_LAYER_SIZE[0],HIDDEN_LAYER_SIZE[1]])
b_fc2 = bias_variable([1,HIDDEN_LAYER_SIZE[1]])

x_fc2 = tf.sigmoid(tf.matmul(x_fc1_drop, W_fc2) + b_fc2) #tf.nn.relu(tf.matmul(x_fc1_drop, W_fc2) + b_fc2)
x_fc2_drop = tf.nn.dropout(x_fc2, keep_prob)
# Finally, we add a softmax layer, the same one if we use just a  
# simple [softmax regression](https://en.wikipedia.org/wiki/Softmax_function).

# readout layer for deep net
W_readout = weight_variable([HIDDEN_LAYER_SIZE[1], labels_count])
b_readout = bias_variable([1,labels_count])

#y = tf.nn.sigmoid_cross_entropy_with_logits(tf.matmul(x_fc1_drop, W_fc2) + b_fc2)
y = tf.sigmoid(tf.matmul(x_fc2_drop, W_readout) + b_readout) #tf.nn.softmax(tf.matmul(x_fc2_drop, W_readout) + b_readout)
if DEBUG:
    names = ['x_flat', 'W_fc1', 'b_fc1', 'x_fc1', 'x_fc1_drop']
    shapes= [x_flat.get_shape(),W_fc1.get_shape(),b_fc1.get_shape(),x_fc1.get_shape(),x_fc1_drop.get_shape()]
    print(list(zip(names,[str(x) for x in shapes])))
    names = ['x_fc1_drop','W_fc2', 'b_fc2', 'x_fc2', 'x_fc2_drop']
    shapes= [x_fc1_drop.get_shape(),W_fc2.get_shape(), b_fc2.get_shape(), x_fc2.get_shape(), x_fc2_drop.get_shape()]
    print(list(zip(names,[str(x) for x in shapes])))
    names = ['x_fc2_drop','W_readout','b_readout','y']
    shapes= [x_fc2_drop.get_shape(),W_readout.get_shape(),b_readout.get_shape(),y.get_shape()]
    print(list(zip(names,[str(x) for x in shapes])))
    print('shape of y :',y.get_shape())
    print('shape of y_:',y_.get_shape())

# To evaluate network performance we use 
# [cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) 
# and to minimise it [ADAM optimiser](http://arxiv.org/pdf/1412.6980v8.pdf) is used. 
# 
# ADAM optimiser is a gradient based optimization algorithm, based 
# on adaptive estimates, it's more sophisticated than steepest 
# gradient descent and is well suited for problems with large 
# data or many parameters.

# cost functions
y  = tf.reshape(y,[-1])
y_ = tf.reshape(y_,[-1]) 
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
ones_y = tf.ones_like(y)
ones_y_ = tf.ones_like(y_)
log_loss = -tf.reduce_mean(y_*tf.log(tf.abs(y)+tf.scalar_mul(EPSILON,ones_y)) + (ones_y_-y_)*tf.log(tf.abs(ones_y-y)+tf.scalar_mul(EPSILON,ones_y)))
#auc_score = -auc(y_,y) #this does not work as it is.
#u = tf.unpack(y ,num=BATCH_SIZE)
#v = tf.unpack(y_,num=BATCH_SIZE)
#if DEBUG:
#    print('unpacked y :',type(u).__name__,len(u),type(u[0]).__name__)
#    print('unpacked y_:',type(v).__name__,len(v),type(v[0]).__name__)

#u = [tf.unpack(x ,num=1) for x  in u]
#v = [tf.unpack(x_,num=1) for x_ in v]
#u = [x for [x] in u]
#v = [x for [x] in v]
#if DEBUG:
#    print('unpacked y :',type(u).__name__,len(u),type(u[0]).__name__)
#    print('unpacked y_:',type(v).__name__,len(v),type(v[0]).__name__)

# optimisation function
#train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(log_loss)
# evaluation
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float')) #original
# accuracy = log_loss

ones_y = tf.ones_like(y)
ones_y_ = tf.ones_like(y_)
#accuracy = -auc(y_,y) # this does not work
accuracy = -tf.reduce_mean(y_*tf.log(tf.abs(y)+tf.scalar_mul(EPSILON,ones_y)) + (ones_y_-y_)*tf.log(tf.abs(ones_y-y)+tf.scalar_mul(EPSILON,ones_y)))
#accuracy = -tf.reduce_mean(y_*tf.log(tf.abs(y)+EPSILON) + (tf.ones_like(y_)-y_)*tf.log(tf.abs(tf.ones_like(y)-y))+EPSILON)
#accuracy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_, name=None))

# To predict values from test data, highest probability is picked from 
# "one-hot vector" indicating that chances of  an image being one of the 
# digits are highest.

# prediction function
# [0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1

predict = tf.scalar_mul(1.0,tf.reshape(y,[-1])) # prediction probabilities
# predict = tf.scalar_mul(1.0, y[:,1])    # prediction probabilities
# predict = tf.identity(y[:,1]) #y[:,1] # alt prediction probabilities
# predict = tf.argmax(y,1) # Or, in this case: [0.4, 0.6] => 1

# *Finally neural network structure is defined and TensorFlow graph is 
# ready for training.*
# ## Train, validate and predict
# #### Helper functions
# 
# Ideally, we should use all data for every step of the training, but 
# that's expensive. So, instead, we use small "batches" of random data. 
# 
# This method is called 
# [stochastic training](https://en.wikipedia.org/wiki/Stochastic_gradient_descent). 
# It is cheaper, faster and gives much of the same result.

epochs_completed = 0
index_in_epoch = 0
num_examples = train_rows.shape[0]

# serve data by batches
def next_batch(batch_size):
    
    global train_rows
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_rows = train_rows[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        if DEBUG:
            print('batch_size,num_examples:',batch_size,num_examples)
        assert batch_size <= num_examples
    end = index_in_epoch
    s = train_labels[start:end].size
    shaped_y = np.reshape(train_labels[start:end],(s,1))
    return train_rows[start:end],shaped_y #train_labels[start:end] 
# Now when all operations for every variable are defined in TensorFlow 
# graph all computations will be performed outside Python environment.
# start TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)
# Each step of the loop, we get a "batch" of data points from the training 
# set and feed it to the graph to replace the placeholders.  In this case, 
# it's:  *x, y* and *dropout.*
# 
# Also, once in a while, we check training accuracy on an upcoming "batch".
# 
# On the local environment, we recommend [saving training progress]
# (https://www.tensorflow.org/versions/master/api_docs/python/state_ops.html#Saver),
# so it can be recovered for further training, debugging or evaluation.
# visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []

display_step=1

for i in range(TRAINING_ITERATIONS):
    #get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)     
    #batch_ys = np.reshape(batch_ys,(batch_ys.size,1))

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
        
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs,y_: batch_ys,keep_prob: 1.0})
        if(VALIDATION_SIZE):
            true_ys = np.reshape(validation_labels[0:BATCH_SIZE],(BATCH_SIZE,1))
            validation_accuracy = accuracy.eval(feed_dict={x:validation_rows[0:BATCH_SIZE],y_: true_ys,keep_prob: 1.0})
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
            validation_accuracies.append(validation_accuracy)
        else:
            print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)
        
        # increase display_step
        if i%(display_step*10) == 0 and i:
            display_step *= 10
    # train on batch
    if DEBUG:
        print(batch_xs.shape,batch_ys.shape)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})
# After training is done, it's good to check accuracy on data that 
# wasn't used in training.
# check final accuracy on validation set  
if(VALIDATION_SIZE):
    true_ys = np.reshape(validation_labels,(validation_labels.size,1))
    validation_accuracy = accuracy.eval(feed_dict={x: validation_rows, y_: true_ys, keep_prob: 1.0})
    print('validation_accuracy => %.4f'%validation_accuracy)
    plt.plot(x_range, train_accuracies,'-b', label='Training')
    plt.plot(x_range, validation_accuracies,'-g', label='Validation')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax = 1.1, ymin = -0.1)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    if GRAPH:
        plt.show()

# predict test set
predicted_labels = predict.eval(feed_dict={x: test_rows, keep_prob: 1.0})

# using batches is more resource efficient
predicted_labels = np.zeros(test_rows.shape[0])
for i in range(0,test_rows.shape[0]//BATCH_SIZE):
    predicted_labels[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={
                       x: test_rows[i*BATCH_SIZE : (i+1)*BATCH_SIZE], keep_prob: 1.0})

if DEBUG:
    print('predicted_labels({0})'.format(len(predicted_labels)))
    print(predicted_labels)

# save results
np.savetxt('submission_'+tag+'.csv', 
           np.c_[test_ids.values,predicted_labels], 
           delimiter=',', 
           header = 'ID,TARGET', 
           comments = '', 
           fmt='%d,%f')                     # change to %d,%f if output is probability

sess.close()




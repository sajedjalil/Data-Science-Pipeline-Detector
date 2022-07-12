import pandas as pd
import numpy as np
from random import sample
from sklearn.metrics import r2_score
import os.path
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

TotalFeatures = 376 # we have '376' features
FeaturesInUse = 150 # number of features used for prediction starting from feature[0]
TotalSampleSize = 4209 # there are total 4209 training samples. 
MiniTestSampleSize = 409 # Lets keep 'MiniTestSampleSize' samples as test data

# subroutine for input data processing
def PreProcessInputData(test_data, train_data) :
    # first lets process columns, apply LabelEncoder to map features to integers
    # thanks to : https://www.kaggle.com/budhiraja/python-pca-regression-baseline-0-5613
    for c in train_data.columns:
        if train_data[c].dtype == 'object':
            lbl = LabelEncoder() 
            lbl.fit(list(train_data[c].values) + list(test_data[c].values)) 
            train_data[c] = lbl.transform(list(train_data[c].values))
            test_data[c] = lbl.transform(list(test_data[c].values))
            # you may normalize values here if you want, but again, it didn't helped me much
            #maxVal = np.amax(pd.concat([test_data[c],train_data[c]], ignore_index=True))
            #minVal = np.amin(pd.concat([test_data[c],train_data[c]], ignore_index=True))
            #train_data[c] = (train_data[c] - minVal + 1)/(maxVal + 1)
            #test_data[c] = (test_data[c] - minVal + 1)/(maxVal + 1)

    y_train_all = train_data['y'].as_matrix();
    y_train_all = np.reshape(y_train_all, (-1, 1));

    x_train_all = train_data.drop(['y','ID'] , axis = 1);
    x_train_all = x_train_all.drop(x_train_all.columns[FeaturesInUse:TotalFeatures], axis = 1); # drop lat few features to keep first 'FeaturesInUse' features
    x_train_all = x_train_all.as_matrix();
    x_train_all = np.reshape(x_train_all, (-1, FeaturesInUse));

    # let's randomly pick 'MiniTestSampleSize' samples and keep them as mini test data
    indices = sample(range(len(y_train_all)),MiniTestSampleSize)
    # our mini test data
    y_mini_test = y_train_all[indices]
    x_mini_test = x_train_all[indices]

    # remaining is out training data
    y_train = np.delete(y_train_all, indices, axis=0)
    x_train = np.delete(x_train_all, indices, axis=0)

    # lets processes master test sample data
    x_test_ID = test_data['ID'].as_matrix();
    x_test = test_data.drop('ID', axis = 1);
    x_test = x_test.drop(x_test.columns[FeaturesInUse:TotalFeatures], axis = 1);
    x_test = x_test.as_matrix();
    x_test = np.reshape(x_test, (-1, FeaturesInUse));

    return (x_train, y_train, x_mini_test, y_mini_test, x_test_ID, x_test)
    
# read input files
test_data=pd.read_csv('../input/test.csv')
train_data=pd.read_csv('../input/train.csv')

# process input data
# x_data, y_data : our training set
# x_mini_test,  and y_mini_test are our mini test samples to test convergence during training
# test_data_ID and test_data are to be used predict y
(x_data, y_data, x_mini_test, y_mini_test, test_data_ID, x_test) = PreProcessInputData(test_data, train_data)

# now we can refer one row of input features as x_data[0], x_data[1]
x = tf.placeholder(tf.float32, [None, FeaturesInUse]);

# this is our model
W = tf.Variable(tf.truncated_normal(shape=[FeaturesInUse,1], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[1]))
y = tf.matmul(x, W) + b; # predicted values
y_ = tf.placeholder(tf.float32, [None, 1]); # true y values

cross_entropy = tf.reduce_mean(tf.square(y - y_)); # this is out cost function
# alternatively, we can define error function to be R^2, but it didn't helped me
#total_error = tf.reduce_sum(tf.square(tf.subtract(y_, tf.reduce_mean(y_))))
#unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_, y)))
#R_squared = tf.subtract(1.0, tf.div(total_error, unexplained_error))
#cross_entropy = R_squared

train_step = tf.train.AdamOptimizer().minimize(cross_entropy); # lets use adam optimizer

sess = tf.InteractiveSession(); # start the session
tf.global_variables_initializer().run();

# Trainning loop
for j in range(2000):
    for i in range( int((TotalSampleSize-MiniTestSampleSize)/200) ): # lets take 200 samples at time
        batch_x = x_data[i*200:i*200+200]
        batch_y = y_data[i*200:i*200+200]
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
    # let's print prediction error for our mini test data
    if 0 == (j%10) :
        # test error
        y_pred = sess.run(y, feed_dict={x: x_mini_test})
        TestError = r2_score(y_mini_test, y_pred)
        # training error
        y_pred = sess.run(y, feed_dict={x: x_data})
        TrainError = r2_score(y_data, y_pred)
        # print
        print("for iteration : {:06d}".format(j), " <TestErr> : {:10.4f}".format(TestError), " <TrainError> : {:10.4f}".format(TrainError))
    # you might want to break loop here by some means...say you found test error starts increasing

# Test trained model now
y_pred = sess.run(y, feed_dict={x: x_test})

# store to submist.csv
sub = pd.DataFrame()
sub['ID'] = test_data_ID
sub['y'] = y_pred
sub.to_csv('submit.csv', index=False)



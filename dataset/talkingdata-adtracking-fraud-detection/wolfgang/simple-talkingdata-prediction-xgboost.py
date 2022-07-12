import numpy as np
import pandas as pd
import math
import time
from sklearn.neural_network import MLPClassifier
from scipy.sparse import csr_matrix, hstack
import tensorflow as tf
import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

# all attributed = 0 means 0.5 score
# Number of unique os < 500
# Number of unique devices 

def print_duration (start_time, msg):
    print("[%d] %s" % (int(time.time() - start_time), msg))
    start_time = time.time()
    return start_time


def main():
    start_time = time.time()
    train = pd.read_csv('../input/train.csv', skiprows=0, nrows=1000000)
    
    print("Os count: " + str(train['os'].value_counts().count())) 
    print("Device count: " + str(train['device'].value_counts().count())) 
    print("App count: " + str(train['app'].value_counts().count()))
    print("Ip count: " + str(train['ip'].value_counts().count())) 
    print("channel count: " + str(train['channel'].value_counts().count())) 
    
    ### prepare model
    config = tf.ConfigProto(intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        ks.backend.set_session(sess)
        model = Sequential()
        model.add(Dense(30, input_shape=(5,), kernel_initializer ='uniform', activation='relu'))
        model.add(Dropout(0.50))
        model.add(Dense(30, activation='relu'))
        model.add(Dropout(0.50))
        model.add(Dense(30, activation='relu'))
        model.add(Dropout(0.50))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        
        train_X = train.as_matrix(columns=['ip', 'app', 'device', 'os', 'channel'])
        model.fit(x=train_X, y=train_X['is_attributed'])
        print(model.evaluate(x=train_X, y=train_X['is_attributed'], verbose=0))
        start_time = print_duration (start_time, "Finished training, start prediction")    
        # read test data set
        test = pd.read_csv('../input/test.csv')
        test_X = test.as_matrix(columns=['ip', 'app', 'device', 'os', 'channel'])
        pred = model.predict(test_X)
        start_time = print_duration (start_time, "Finished prediction, start store results")    
        submission = pd.read_csv("../input/sample_submission.csv")
        submission['is_attributed'] = pred
        submission.to_csv("submission.csv", index=False)
        start_time = print_duration(start_time, "Finished to store result")
    
    
if __name__ == '__main__':
    main()
    
    
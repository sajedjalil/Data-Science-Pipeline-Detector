# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:40:37 CEST 2015

@author: cerberusd
simple starting script, without the use of MNE
Thanks to @author: Elena Cuoco 


"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from glob import glob
import os

from sklearn.preprocessing import StandardScaler

from lasagne import nonlinearities
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet 

#############function to read data###########

def prepare_data_train(fname):
    """ read and prepare training data """
    # Read data
    data = pd.read_csv(fname)
    # events file
    events_fname = fname.replace('_data','_events')
    # read event file
    labels= pd.read_csv(events_fname)
    clean=data.drop(['id' ], axis=1)#remove id
    labels=labels.drop(['id' ], axis=1)#remove id
    return  clean,labels

def prepare_data_test(fname):
    """ read and prepare test data """
    # Read data
    data = pd.read_csv(fname)
    return data

scaler= StandardScaler()
def data_preprocess_train(X):
    X_prep=scaler.fit_transform(X)
    #do here your preprocessing
    return X_prep
def data_preprocess_test(X):
    X_prep=scaler.transform(X)
    #do here your preprocessing
    return X_prep

##downsamplig naive like this is not correct, if you do not low pass filter.
##this down sampling here it needed only to keep the script run below 10 minutes.
## please do not downsample or use correct procedure to decimate data without alias
subsample=100 # training subsample.if you want to downsample the training data
#######columns name for labels#############
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']




#######number of subjects###############
subjects = range(1,13)
ids_tot = []
pred_tot = []
nnpred_tot = []

net1 = NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape=(None, 32),  # 96x96 input pixels per batch
        hidden_num_units=50,  # number of units in hidden layer
        output_nonlinearity=nonlinearities.sigmoid,  # output layer uses identity function
        output_num_units=1,  # 1 target value
    
        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
    
        regression=True,  # flag to indicate we're dealing with regression problem
        max_epochs=50,  # we want to train this many epochs
        verbose=1,
    )

#reshape(-1,1, 14224, 32)
net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv1DLayer),
        ('conv2', layers.Conv1DLayer),
        ('pool1', layers.MaxPool1DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 32, 1),
    conv1_num_filters=4, conv1_filter_size=1, 
    conv2_num_filters=8, conv2_filter_size=5, conv2_border_mode='same', pool1_pool_size=4,
    hidden4_num_units=50, hidden5_num_units=50,
    output_num_units=1, output_nonlinearity=nonlinearities.sigmoid,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=50,
    verbose=1,
    )



###loop on subjects and 8 series for train data + 2 series for test data
for subject in subjects:
    y_raw= []
    raw = []
    ################ READ DATA ################################################
    fnames =  glob('../input/train/subj%d_series*_data.csv' % (subject))
    for fname in fnames:
      data,labels=prepare_data_train(fname)
      raw.append(data)
      y_raw.append(labels)

    X = pd.concat(raw)
    y = pd.concat(y_raw)
    #transform in numpy array
    #transform train data in numpy array
    X_train =np.asarray(X.astype(float))
    y = np.asarray(y.astype(float))


    ################ Read test data #####################################
    #
    fnames =  glob('../input/test/subj%d_series*_data.csv' % (subject))
    test = []
    idx=[]
    for fname in fnames:
      data=prepare_data_test(fname)
      test.append(data)
      idx.append(np.array(data['id']))
    X_test= pd.concat(test)
    ids=np.concatenate(idx)
    ids_tot.append(ids)
    X_test=X_test.drop(['id' ], axis=1)#remove id
    #transform test data in numpy array
    X_test =np.asarray(X_test.astype(float))


    ################ Train classifiers ########################################
    
    
    
    
    
    
    lr = LogisticRegression()
    pred = np.empty((X_test.shape[0],6))
    nnpred = np.empty((X_test.shape[0],6))
    X_train=data_preprocess_train(X_train)
    X_test=data_preprocess_test(X_test)
    for i in range(6):
        y_train= y[:,i]
        print('Train subject %d, class %s' % (subject, cols[i]))
        lr.fit(X_train[::subsample,:],y_train[::subsample])    
        net2.fit(X_train[::subsample,:].reshape(-1, 32, 1).astype('float32').astype('float32'), y_train[::subsample].astype('float32'))
        nnpred[:,i] = net2.predict(X_test.reshape(-1, 32, 1)).ravel()
        pred[:,i] = lr.predict_proba(X_test)[:,1]

    pred_tot.append(pred)
    nnpred_tot.append(nnpred)

    
    
# submission file
submission_file = 'grasp-sub-simple.csv'
# create pandas object for sbmission
submission = pd.DataFrame(index=np.concatenate(ids_tot),
                          columns=cols,
                          data=np.concatenate(nnpred_tot))

# write file
submission.to_csv(submission_file,index_label='id',float_format='%.3f')
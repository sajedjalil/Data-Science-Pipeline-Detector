# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:40:37 CEST 2015

@author: Elena Cuoco
simple starting script, without the use of MNE
Thanks to @author: alexandrebarachant for his wornderful starting script


"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from glob import glob
import scipy
from scipy.signal import butter, lfilter, convolve, boxcar
from scipy.signal import freqz
from scipy.fftpack import fft, ifft
import os

from sklearn.preprocessing import StandardScaler


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def smooth(x,window_len=11,window='hanning'):

    if window_len<3:
        return x



    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y
 
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
# training subsample.if you want to downsample the training data
subsample = 10
#######columns name for labels#############
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

############ START Beat the Benchmark. 0.67+
# design a butterworth bandpass filter 
freqs = [7, 30]
b,a = butter(5,np.array(freqs)/250.0,btype='bandpass')

# CSP parameters
# Number of spatial filter to use
nfilters = 4

# convolution
# window for smoothing features
nwin = 250

############ END Beat the Benchmark. 0.67+


#######number of subjects###############
#subjects = range(1,13)
subjects = range(1,2)
ids_tot = []
pred_tot = []

###loop on subjects and 8 series for train data + 2 series for test data
for subject in subjects:
    y_raw= []
    raw = []
    ################ READ DATA ################################################
    fnames =  glob('../input/train/subj%d_series*_data.csv' % (subject))
#    fnames =  glob('../input/train/subj1_series1_events.csv')
#    fnames =  glob('../input/train/subj1_series1_data.csv')
    for fname in fnames:
      data,labels=prepare_data_train(fname)
      raw.append(data)
      y_raw.append(labels)

    for fname in fnames:
      with open(fname) as myfile:
        head = [next(myfile) for x in range(10)]
      print(head)
        
    X = pd.concat(raw)
    y = pd.concat(y_raw)
    #transform in numpy array
    #transform train data in numpy array
    X_train =np.asarray(X.astype(float))
    y_train_raw = np.asarray(y.astype(float))


#    plt.plot(X_train)
#    plt.savefig('X_train.png')
#    plt.close();
#    print(X_train.shape)
    X_train_smooth = []

#    for i in range(0,32):
#        X_train_smooth = smooth(X_train[:,i],10000)
#        print(X_train_smooth.shape)
#        plt.plot(X_train_smooth)
#    plt.savefig('X_train_smooth.png')
#    plt.close();
    X_train_con = np.mean(X_train, axis=1)
#    print(X_train.shape)
#    print(X_train_con.shape)
#    plt.plot(X_train_con)
#    plt.show();
#    plt.savefig('X_train_con.png')
#    plt.close()
    X_train_con_smooth = smooth(X_train_con,10000);
#    plt.plot(X_train_con_smooth)
#    plt.show();
#    plt.savefig('X_train_con_smooth')
#    plt.close()
    
        # Sample rate and desired cutoff frequencies (in Hz).
    fs = 500.0
    lowcut = 15.0
    highcut = 20.0

    y = butter_bandpass_filter(X_train_con, lowcut, highcut, fs, order=6)
#   plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.plot(y, label='Filtered signal (%g Hz)' % 500)
#    plt.xlabel('time (seconds)')
#    plt.hlines([-a, a], 0, T, linestyles='--')
#    plt.grid(True)
#    plt.axis('tight')
#    plt.legend(loc='upper left')

    plt.show()
    plt.savefig('butterworth.png')
    plt.close()



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
    X_train=data_preprocess_train(X_train)
    X_test=data_preprocess_test(X_test)
    print(X_train_con.shape)
    print("sijunfejuin")
    X_train_con.reshape(1422392,1)
    for i in range(6):
        y_train= y_train_raw[:,i]
        print('Train subject %d, class %s' % (subject, cols[i]))
#        lr.fit(X_train[::subsample,:],y_train[::subsample])
        lr.fit(X_train_con,y_train)
        pred[:,i] = lr.predict_proba(X_test)[:,1]

    pred_tot.append(pred)

# submission file
submission_file = 'grasp-sub-simple.csv'
# create pandas object for sbmission
submission = pd.DataFrame(index=np.concatenate(ids_tot),
                          columns=cols,
                          data=np.concatenate(pred_tot))

# write file
submission.to_csv(submission_file,index_label='id',float_format='%.3f')


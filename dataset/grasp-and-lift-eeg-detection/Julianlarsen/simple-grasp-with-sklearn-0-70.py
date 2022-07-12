# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:40:37 CEST 2015

@author: Elena Cuoco
simple starting script, without the use of MNE
Thanks to @author: alexandrebarachant for his wornderful starting script


"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from glob import glob
import os

from sklearn.preprocessing import StandardScaler
 
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
#######columns name for labels#############
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

#######number of subjects###############
subjects = range(1,4) # 13 subjects
ids_tot = []
pred_tot = []

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
    
    lag_order = 2
    
    err = np.empty((X_test.shape[0],6))
    err[0:lag_order,:] = 0
    X_train=data_preprocess_train(X_train)
    X_test=data_preprocess_test(X_test)
    models = []
    
    for i in range(6):
        print('Training subject %d, class %s' % (subject, cols[i]))
        # select timeseries for state i
        t = X_train[y[:,i]>0,:]
        # fit model 
        models.append(sm.tsa.VAR(t).fit(lag_order))
        
        for n in range(lag_order+1,X_test.shape[0]-1):
            pt = models[i].forecast(X_test[n-lag_order:n,:], 1)
            err[n,i]=np.linalg.norm(pt-X_test[n+1,:])
        
    
    # normalize errors in pred
    pred = np.divide(1,err)
    pred[0:lag_order+1,:] = 0

    for n in range(lag_order+1,pred.shape[0]-1):
        pred[n,:] = pred[n,:]*(1/(sum(pred[n,:])))
        
    pred_tot.append(pred)

# submission file
submission_file = 'grasp-sub-simple.csv'
# create pandas object for sbmission
submission = pd.DataFrame(index=np.concatenate(ids_tot),
                          columns=cols,
                          data=np.concatenate(pred_tot))

# write file
submission.to_csv(submission_file,index_label='id',float_format='%.3f')

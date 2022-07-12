# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:00:37 2015

@author: alexandrebarachant

Beat the benchmark with CSP and Logisitic regression.

General Idea :

The goal of this challenge is to detect events related to hand movements. Hand 
movements are caracterized by change in signal power in the mu (~10Hz) and beta
(~20Hz) frequency band over the sensorimotor cortex. CSP spatial filters are
trained to enhance signal comming from this brain area, instantaneous power is
extracted and smoothed, and then feeded into a logisitic regression.

Preprocessing :

Signal are bandpass-filtered between 7 and 30 Hz to catch most of the signal of
interest. 4 CSP spatial filter are then applied to the signal, resutlting to
4 new time series.  In order to train CSP spatial filters, EEG are epoched 
using a window of 1.5 second before and after the event 'Replace'. CSP training
needs two classes. the epochs before Replace event are assumed to contain 
patterns corresponding to hand movement, and epochs after are assumed to 
contain resting state.

Feature extraction :

Preprocessing is applied, spatialy filtered signal are the rectified and 
convolved with a 0.5 second rectangular window for smoothing. Then a logarithm
is applied. the resutl is a vector of dimention 4 for each time sample.

Classification :

For each of the 6 event type, a logistic regression is trained. For training 
only, features are downsampled in oder to speed up the process. Prediction are
the probailities of the logistic regression.

"""

print(__doc__)

import numpy as np
import pandas as pd
from mne.io import RawArray
from mne.channels import read_montage
from mne.epochs import concatenate_epochs
from mne import create_info, find_events, Epochs, concatenate_raws, pick_types
from mne.decoding import CSP

from sklearn.linear_model import LogisticRegression
from glob import glob

from scipy.signal import butter, lfilter, convolve, boxcar
from joblib import Parallel, delayed

def creat_mne_raw_object(fname,read_events=True):
    """Create a mne raw instance from csv file"""
    # Read EEG file
    data = pd.read_csv(fname)
    
    # get chanel names
    ch_names = list(data.columns[1:])
    
    # read EEG standard montage from mne
    montage = read_montage('standard_1005',ch_names)

    ch_type = ['eeg']*len(ch_names)
    data = 1e-6*np.array(data[ch_names]).T
    
    if read_events:
        # events file
        ev_fname = fname.replace('_data','_events')
        # read event file
        events = pd.read_csv(ev_fname)
        events_names = events.columns[1:]
        events_data = np.array(events[events_names]).T
        
        # define channel type, the first is EEG, the last 6 are stimulations
        ch_type.extend(['stim']*6)
        ch_names.extend(events_names)
        # concatenate event file and data
        data = np.concatenate((data,events_data))
        
    # create and populate MNE info structure
    info = create_info(ch_names,sfreq=500.0, ch_types=ch_type, montage=montage)
    info['filename'] = fname
    
    # create raw object 
    raw = RawArray(data,info,verbose=False)
    
    return raw

subjects = range(1,13)
ids_tot = []
pred_tot = []
all_train = []
all_test = []

# design a butterworth bandpass filter 
freqs = [7, 30]
b,a = butter(5,np.array(freqs)/250.0,btype='bandpass')

# CSP parameters
# Number of spatial filter to use
nfilters = 4

# convolution
# window for smoothing features
nwin = 250

# training subsample
subsample = 10

# submission file
submission_file = 'beat_the_benchmark.csv'
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

for subject in subjects:
    epochs_tot = []
    y = []

    ################ READ DATA ################################################
    fnames =  glob('../input/train/subj%d_series*_data.csv' % (subject))
    
    # read and concatenate all the files
    raw = concatenate_raws([creat_mne_raw_object(fname) for fname in fnames])
       
    # pick eeg signal
    picks = pick_types(raw.info,eeg=True)
    
    # Filter data for alpha frequency and beta band
    # Note that MNE implement a zero phase (filtfilt) filtering not compatible
    # with the rule of future data.
    # Here we use left filter compatible with this constraint. 
    # The function parallelized for speeding up the script
    raw._data[picks] = np.array(Parallel(n_jobs=-1)(delayed(lfilter)(b,a,raw._data[i]) for i in picks))
    
    ################ CSP Filters training #####################################
    # get event posision corresponding to Replace
    events = find_events(raw,stim_channel='Replace', verbose=False)
    # epochs signal for 1.5 second before the movement
    epochs = Epochs(raw, events, {'during' : 1}, -2, -0.5, proj=False,
                    picks=picks, baseline=None, preload=True,
                    add_eeg_ref=False, verbose=False)
    
    epochs_tot.append(epochs)
    y.extend([1]*len(epochs))
    
    # epochs signal for 1.5 second after the movement, this correspond to the 
    # rest period.
    epochs_rest = Epochs(raw, events, {'after' : 1}, 0.5, 2, proj=False,
                    picks=picks, baseline=None, preload=True,
                    add_eeg_ref=False, verbose=False)
    
    # Workaround to be able to concatenate epochs with MNE
    epochs_rest.times = epochs.times
    
    y.extend([-1]*len(epochs_rest))
    epochs_tot.append(epochs_rest)
        
    # Concatenate all epochs
    epochs = concatenate_epochs(epochs_tot)
    
    # get data 
    X = epochs.get_data()
    y = np.array(y)
    
    # train CSP
    csp = CSP(n_components=nfilters, reg='lws')
    csp.fit(X,y)
    
    ################ Create Training Features #################################
    # apply csp filters and rectify signal
    feat = np.dot(csp.filters_[0:nfilters],raw._data[picks])**2
    
    # smoothing by convolution with a rectangle window    
    feattr = np.array(Parallel(n_jobs=-1)(delayed(convolve)(feat[i],boxcar(nwin),'full') for i in range(nfilters)))
    feattr = np.log(feattr[:,0:feat.shape[1]])
    feattr = pd.DataFrame(feattr)

    feattr['subject'] =subject
    all_train.append(feattr)
   
    ################ Create test Features #####################################
    # read test data 
    fnames =  glob('../input/test/subj%d_series*_data.csv' % (subject))
    raw = concatenate_raws([creat_mne_raw_object(fname, read_events=False) for fname in fnames])
    raw._data[picks] = np.array(Parallel(n_jobs=-1)(delayed(lfilter)(b,a,raw._data[i]) for i in picks))
    
    # read ids
    ids = np.concatenate([np.array(pd.read_csv(fname)['id']) for fname in fnames])
    ids_tot.append(ids)
    
    # apply preprocessing on test data
    feat = np.dot(csp.filters_[0:nfilters],raw._data[picks])**2
    featte = np.array(Parallel(n_jobs=-1)(delayed(convolve)(feat[i],boxcar(nwin),'full') for i in range(nfilters)))
    featte = np.log(featte[:,0:feat.shape[1]])
    featte = pd.DataFrame(featte)
    featte['subject'] =subject
    all_test.append(featte)

s_data = pd.DataFrame(all_train)
t_data = pd.DataFrame(all_test)

s_data.to_csv("train_alex.csv", delimiter=",")
t_data.to_csv("test_alex.csv",  delimiter=",")
   
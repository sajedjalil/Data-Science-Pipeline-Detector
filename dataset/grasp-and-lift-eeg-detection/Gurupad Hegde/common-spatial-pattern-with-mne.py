# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:00:37 2015

@author: alexandrebarachant

During a hand movement, the mu (~10Hz) and beta (~20Hz) oscillations are suppressed 
over the contralateral motor cortex, i.e. we can observe a reduction of the 
signal power in the corresponding frequency band. This effect is know as 
Event Related Desynchronization.

I used MNE python to epoch signal corresponding to the hand movement, by assuming that 
the hand movement occur before the 'Replace' event.

Using Common spatial patterns algorithm, i extract spatial filters that maximize 
the difference of variance during and after the movement, and then visualize the 
corresponding spectrum. 

For each subject, we should see a spot over the electrode C3 (Left motor cortex,
corresponding to a right hand movement), and a decrease of the signal power in 
10 and 20 Hz during the movement (by reference to after the movement).

Each subject has a different cortex organization, and a different apha and beta 
peak. The CSP algorithm is also sensitive to artefacts, so it could give eronous 
maps (for example subject 5 seems to trig on eye movements)

"""

import numpy as np
import pandas as pd
from mne.io import RawArray
from mne.channels import read_montage
from mne.epochs import concatenate_epochs
from mne import create_info, find_events, Epochs
from mne.viz.topomap import _prepare_topo_plot, plot_topomap
from mne.decoding import CSP

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import cross_val_score, LeaveOneLabelOut
from glob import glob

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.signal import welch
from mne import pick_types

def creat_mne_raw_object(fname):
    """Create a mne raw instance from csv file"""
    # Read EEG file
    data = pd.read_csv(fname)
    
    # get chanel names
    ch_names = list(data.columns[1:])
    
    # read EEG standard montage from mne
    montage = read_montage('standard_1005',ch_names)

    # events file
    ev_fname = fname.replace('_data','_events')
    # read event file
    events = pd.read_csv(ev_fname)
    events_names = events.columns[1:]
    events_data = np.array(events[events_names]).T
    
    # concatenate event file and data
    data = np.concatenate((1e-6*np.array(data[ch_names]).T,events_data))        
    
    # define channel type, the first is EEG, the last 6 are stimulations
    ch_type = ['eeg']*len(ch_names) + ['stim']*6
    
    # create and populate MNE info structure
    ch_names.extend(events_names)
    info = create_info(ch_names,sfreq=500.0, ch_types=ch_type, montage=montage)
    info['filename'] = fname
    
    # create raw object 
    raw = RawArray(data,info,verbose=False)
    return raw

subjects = range(1,13)
auc = []
for subject in subjects:
    epochs_tot = []
    epochs_tot_test = []
    
    #eid = 'HandStart'
    fnames =  glob('../input/train/subj%d_series*_data.csv' % (subject))
    fnames_test =  glob('../input/test/subj%d_series*_data.csv' % (subject))
    
    session = []
    session_test = []
    y = []
    for i,fname in enumerate(fnames):
      
        # read data 
        raw = creat_mne_raw_object(fname)
        raw_test = creat_mne_raw_object(fname)
        
        # pick eeg signal
        picks = pick_types(raw.info,eeg=True)
        picks_test = pick_types(raw_test.info,eeg=True)
        
        # Filter data for alpha frequency and beta band
        # Note that MNE implement a zero phase (filtfilt) filtering not compatible
        # with the rule of future data.
        raw.filter(7,35, picks=picks, method='iir', n_jobs=-1, verbose=False)
        raw_test.filter(7,35, picks=picks_test, method='iir', n_jobs=-1, verbose=False)
        
        # get event posision corresponding to Replace
        events = find_events(raw,stim_channel='Replace', verbose=False)
        events_test = find_events(raw_test,stim_channel='Replace', verbose=False)
        
        # epochs signal for 1.5 second before the movement
        epochs = Epochs(raw, events, {'during' : 1}, -2, -0.5, proj=False,
                        picks=picks, baseline=None, preload=True,
                        add_eeg_ref=False, verbose=False)
        epochs_test = Epochs(raw_test, events, {'during' : 1}, -2, -0.5, proj=False,
                        picks=picks_test, baseline=None, preload=True,
                        add_eeg_ref=False, verbose=False)
        
        epochs_tot.append(epochs)
        epochs_tot_test.append(epochs_test)
        
        session.extend([i]*len(epochs))
        session_test.extend([i]*len(epochs_test))
        
        y.extend([1]*len(epochs))
        
        # epochs signal for 1.5 second after the movement, this correspond to the 
        # rest period.
        epochs_rest = Epochs(raw, events, {'after' : 1}, 0.5, 2, proj=False,
                        picks=picks, baseline=None, preload=True,
                        add_eeg_ref=False, verbose=False)
                        
        epochs_rest_test = Epochs(raw_test, events_test, {'after' : 1}, 0.5, 2, proj=False,
                        picks=picks_test, baseline=None, preload=True,
                        add_eeg_ref=False, verbose=False)
        
        # Workaround to be able to concatenate epochs
        epochs_rest.times = epochs.times
        epochs_rest_test.times = epochs_test.times
        
        epochs_tot.append(epochs_rest)
        epochs_tot_test.append(epochs_rest_test)
        
        session.extend([i]*len(epochs_rest))
        session_test.extend([i]*len(epochs_rest_test))
        
        y.extend([-1]*len(epochs_rest))
        
    #concatenate all epochs
    epochs = concatenate_epochs(epochs_tot)
    epochs_test = concatenate_epochs(epochs_tot_test)
    
    # get data 
    X = epochs.get_data()
    y = np.array(y)
    
    X_test = epochs_test.get_data()
    
    # run CSP
    csp = CSP(reg='lws')
    #csp.fit(X,y)
    
    # compute spatial filtered spectrum
    # po = []
    # for x in X:
    #     f,p = welch(np.dot(csp.filters_[0,:].T,x), 500, nperseg=512)
    #     po.append(p)
    # po = np.array(po)
    
    # run cross validation
    clf = make_pipeline(csp,LogisticRegression())
    cv = LeaveOneLabelOut(session)
    auc.append(cross_val_score(clf,X,y,cv=cv,scoring='roc_auc').mean())
    print("Subject %d : AUC cross val score : %.3f" % (subject,auc[-1]))
    clf.fit(X,y)
    preds = clf.predict(X_test)
    print(preds)
    


#preds.to_csv('submission.csv')

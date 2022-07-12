# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:37:26 2015

@author: alexandrebarachant

This script illustrates the presence of a VEP associated with the led lighting 
ON when the subject is cued to start the movement.

the html part has been inspired from the  Normalized Kaggle Distance for 
visualization from Triskelion.
"""


import numpy as np
import pandas as pd
from mne.io import RawArray
from mne.channels import read_montage
from mne import create_info, find_events, Epochs, concatenate_raws, pick_types
from mne import compute_raw_data_covariance
from mne.viz import plot_image_epochs, plot_topomap
from mne.viz import plot_topomap, plot_topo
from scipy.linalg import eigh, inv
import matplotlib.pyplot as plt

from glob import glob

def fit_xdawn(evoked, signal_cov):
    """Minimal implementation of xdawn."""
    cov_evoked = np.cov(evoked)
    evals, evecs = eigh(cov_evoked, signal_cov)
    evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
    evecs /= np.sqrt(np.sum(evecs ** 2, axis=0))
    A = inv(evecs.T)
    return evecs, A

def apply_xdawn(epochs, V, A, n_components=3):
    """Xdawn denoising."""
    data = epochs.get_data()
    sources = np.dot(V.T, data).transpose((1,0,2))
    sources[:,n_components:,:] = 0
    data = np.dot(A, sources).transpose((1,0,2))
    epochs._data = data
    return epochs

def creat_mne_raw_object(fname, read_events=True):
    """Create a mne raw instance from csv file."""
    # Read EEG file
    data = pd.read_csv(fname)

    # get chanel names
    ch_names = list(data.columns[1:])

    # read EEG standard montage from mne
    montage = read_montage('standard_1005', ch_names)

    ch_type = ['eeg']*len(ch_names)
    data = 1e-6*np.array(data[ch_names]).T

    if read_events:
        # events file
        ev_fname = fname.replace('_data', '_events')
        # read event file
        events = pd.read_csv(ev_fname)
        events_names = events.columns[1:]
        events_data = np.array(events[events_names]).T

        # define channel type, the first is EEG, the last 6 are stimulations
        ch_type.extend(['stim']*6)
        ch_names.extend(events_names)
        # concatenate event file and data
        data = np.concatenate((data, events_data))

    # create and populate MNE info structure
    info = create_info(ch_names, sfreq=500.0, ch_types=ch_type,
                       montage=montage)
    info['filename'] = fname

    # create raw object
    raw = RawArray(data, info, verbose=False)

    return raw

subject = 5

fnames =  glob('../input/train/subj%d_series[1-8]_data.csv' % (subject))
# read and concatenate all the files
raw = [creat_mne_raw_object(fname) for fname in fnames]
raw = concatenate_raws(raw)
# pick eeg signal
picks = pick_types(raw.info, eeg=True)    

raw.filter(1, 20, picks=picks)

events = find_events(raw,stim_channel='HandStart')
epochs = Epochs(raw, events, {'HandStart' : 1}, -0.2, 0.6, proj=False,
                picks=picks, baseline=None, preload=True, 
                add_eeg_ref=True, verbose =False)
  
evoked = epochs.average()

evoked.plot(show=False)
plt.savefig('evoked_time.png' ,bbox_inches='tight', dpi=300)

plot_topo(evoked, show=False)
plt.savefig('evoked_topo.png' ,bbox_inches='tight', facecolor='k', dpi=300)

evoked.plot_topomap(times=[-0.1, 0, 0.1, 0.15, 0.25, 0.3], show=False)
plt.savefig('evoked_topomap.png' ,bbox_inches='tight', dpi=300)

plot_image_epochs(epochs, picks=[-3], sigma=5, vmin=-75, vmax=75, show=False)
plt.savefig('epochs_image.png' ,bbox_inches='tight', dpi=300)



#denoising with Xdawn
signal_cov = compute_raw_data_covariance(raw, picks=picks).data

V, A = fit_xdawn(evoked.data, signal_cov)
epochs_dn = apply_xdawn(epochs, V, A, n_components=4)
plot_image_epochs(epochs_dn, picks=[-3], sigma=5, vmin=-70, vmax=70, show=False)
plt.savefig('epochs_xdawn_image.png' ,bbox_inches='tight', dpi=300)


# Generate html file

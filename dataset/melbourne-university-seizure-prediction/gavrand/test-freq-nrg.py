# Borrowed from https://plot.ly/matplotlib/fft/
# Credit to SolverWorld https://www.kaggle.com/c/melbourne-university-seizure-prediction/forums/t/23392/frequency-content-of-a-channel

import sys
import os
import numpy as np
import pandas as pd
from math import *
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

#import pyeeg 
# pyeeg is the one that has very good fractal dimensions 
# computation but not installed here

def mat_to_data(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return ndata

def channel_nrg(path,ch):
    f = mat_to_data(path)
    fs = f['iEEGsamplingRate'][0,0]
    d = f['data']
    #[rows,n] = d.shape
    #print((rows, n))
    y = d[:,ch] # first column 
    n = len(y) # length of the signal
    print(n)
    
    Fs = 400.0;  # sampling rate
    Ts = 1.0/Fs; # sampling interval
    t = np.arange(0,n/Fs,Ts)
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n//2)] # one side frequency range
    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(n//2)]
    fig, ax = plt.subplots(2,1)
    ax[0].plot(t,y)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    #plt.show()
    plt.savefig('fft'+ str(ch+1).zfill(2) +'.png')
    return 0

for i in range(15):
    channel_nrg('../input/train_1/1_145_1.mat',i)


#feat = calculate_features('../input/train_1/1_145_1.mat')
#print(feat)
#print(feat.shape)


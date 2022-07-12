#Based on notebook by Deep: https://www.kaggle.com/deepcnn/melbourne-university-seizure-prediction/spectrogram-pairs/notebook

import datetime
import pandas as pd
import numpy as np
from scipy.io import loadmat
from operator import itemgetter
import random
import os
import time
import glob
import math
from matplotlib import pyplot as plt
from pylab import plot, show, subplot, specgram, imshow, savefig

N = 16

def mat_to_dataframe(path):
    print("Processing %s" % path)
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    log_std = lambda x: math.log(np.std(x))
    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0]).rolling(10000).apply(log_std).iloc[9999::10000, :]
    
def plot_spectrogram(pairs):
    l = []
    l.extend(range(0, 144, 24))

    d0 = mat_to_dataframe(pairs[0][0])
    d1 = mat_to_dataframe(pairs[0][1])
    for pair in pairs[1:]:
        d0 = d0.append(mat_to_dataframe(pair[0]))
        d1 = d1.append(mat_to_dataframe(pair[1]))
    
    X0 = d0.as_matrix()
    d0.to_csv('class0.csv')
    print("Shape of d0: {}".format(d0.shape))
    X1 = d1.as_matrix()
    d1.to_csv('class1.csv')
    print("Shape of d1: {}".format(d1.shape))
    
    for i in range(N):         
        plt.subplot(2, 1, 1)
        plt.plot(X0[:,i])
        for j in l:
            plt.axvline(j, linewidth=0.5, color='r')
        plt.title('ch ' + str(i) + ': class 0 - log_std, rolling by 10K')
        
        plt.subplot(2, 1, 2)
        plt.plot(X1[:,i])
        for j in l:
            plt.axvline(j, linewidth=0.5, color='r')
        plt.title('ch ' + str(i) + ': class 1 - log_std, rolling by 10K')
        
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, hspace=0.8)
        savefig('ch_' + str(i) + ".png")
        plt.gcf().clear()

pairs = []
start = 1; stop = 3
for i in range(start,stop):
    pairs.append(['../input/train_1/1_' + str(i) + '_0.mat', 
                  '../input/train_1/1_' + str(i) + '_1.mat'])

plot_spectrogram(pairs)
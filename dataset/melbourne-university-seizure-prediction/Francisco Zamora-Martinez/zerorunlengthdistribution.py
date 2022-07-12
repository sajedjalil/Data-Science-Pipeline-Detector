import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.io
import scipy.stats
import sys

from glob import glob
from pylab import savefig

def computeZeroRunLenghts(subjects):
    labels  = { k:[] for k in subjects }
    runlens = { k:[] for k in subjects }
    for subj in subjects:
        print("# Counting zero run lengths of", subj)
        files = sorted(glob("../input/" + subj + "/*.mat"))
        for i,filename in enumerate(files):
            # if i % 100 == 0: print("%3.0f %%"%(float(i)/len(files)*100.0))
            try:
                # Corrupted files can be loaded using parameter: verify_compressed_data_integrity=False
                x = sp.io.loadmat(filename)["dataStruct"]
                data = x["data"].item(0).astype(np.float32)
                z = (np.sum(np.abs(data),1) > 0).astype(sp.int8)
                bounded = np.hstack([[1], z, [1]])
                d = np.diff(bounded)
                starts, = np.where(d < 0)
                stops, = np.where(d > 0)
                current_runlen = stops - starts
                runlens[subj].append(current_runlen)
                labels[subj].append(int(filename[-5:-4]))
            except:
                print(filename, "is corrupted")
    runlens = { k:np.array(v) for k,v in runlens.items() }
    labels  = { k:np.array(v) for k,v in labels.items() }
    return runlens,labels

def plotHistogram(ax, title, x):
    ax.set(xlabel="Zeros run length (x1000)", ylabel="Relative frequency")
    ax.set_ylim(0,1.0)
    ax.set_xlim(0,240)
    ax.set_title(title)
    n, bins, patches = ax.hist(x/1000.0, 50, normed=False, facecolor='green', alpha=0.75, range=[0,240])
    for item in patches: item.set_height(item.get_height()/(len(x)+1e-10))


subjects = [ "train_1", "train_2", "train_3", "test_1", "test_2", "test_3" ]
runlens,labels = computeZeroRunLenghts(subjects)

for subj in subjects:
    cls_0_runlens = runlens[subj][labels[subj]==0]
    cls_1_runlens = runlens[subj][labels[subj]==1]
    # Compute accumulated percentage of files with at more than % dropout rows
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
    f.suptitle("Folder " + subj)
    plotHistogram(ax1, "label-0", np.hstack(cls_0_runlens))
    plotHistogram(ax2, "label-1", np.hstack(cls_1_runlens))
    savefig(subj + ".png")
    plt.gcf().clear()

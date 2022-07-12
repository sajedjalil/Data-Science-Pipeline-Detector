import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.io
import scipy.stats
import sys

from glob import glob
from pylab import savefig

def dropoutCounts(subjects):
    labels = { k:[] for k in subjects }
    counts = { k:[] for k in subjects }
    for subj in subjects:
        print("# Counting dropouts of", subj)
        files = sorted(glob("../input/" + subj + "/*.mat"))
        for i,filename in enumerate(files):
            # if i % 100 == 0: print("%3.0f %%"%(float(i)/len(files)*100.0))
            try:
                # Corrupted files can be loaded using parameter: verify_compressed_data_integrity=False
                x = sp.io.loadmat(filename)["dataStruct"]
                data = x["data"].item(0).astype(np.float32)
                counts[subj].append( data.shape[0] - np.count_nonzero(np.sum(np.abs(data), 1)) )
                labels[subj].append(int(filename[-5:-4]))
            except:
                print(filename, "is corrupted")
    counts = { k:np.array(v) for k,v in counts.items() }
    labels = { k:np.array(v) for k,v in labels.items() }
    return counts,labels

def printSubjectSummary(prefix, subject_counts):
    n_files = float(len(subject_counts))
    n_dropouts = np.sum(subject_counts > 0)
    print(prefix, n_dropouts, "of", n_files, " (", n_dropouts/n_files*100.0, "% )" )    
    return n_files,n_dropouts

def computeAccumulatedPercentages(subj_counts):
    uniq_values,uniq_counts = np.unique(subj_counts, return_counts=True)
    x = uniq_values[::-1]/240000.0
    y = np.cumsum(uniq_counts[::-1])/float(np.sum(uniq_counts))
    return x,y

def plotPercentages(ax, title, x, y):
    # ax.fill_between(x, 0, y)
    ax.set(xlabel="More than % rows are dropouts", ylabel="% of files")
    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.0)
    ax.set_title(title)
    ax.invert_xaxis()
    ax.step(x, y, linewidth=4, color='r', where="post")

subjects = [ "train_1", "train_2", "train_3", "test_1", "test_2", "test_3" ]
counts,labels = dropoutCounts(subjects)

print("# Number and percentage of files with dropouts")
for subj in subjects:
    cls_0_counts = counts[subj][labels[subj]==0]
    cls_1_counts = counts[subj][labels[subj]==1]
    printSubjectSummary(subj + " label-0", cls_0_counts)
    printSubjectSummary(subj + " label-1", cls_1_counts)
    # Compute accumulated percentage of files with at more than % dropout rows
    x_0,y_0 = computeAccumulatedPercentages(cls_0_counts)
    x_1,y_1 = computeAccumulatedPercentages(cls_1_counts)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    f.suptitle("Folder " + subj)
    plotPercentages(ax1, "label-0", x_0, y_0)
    plotPercentages(ax2, "label-1", x_1, y_1)
    savefig(subj + ".png")
    plt.gcf().clear()
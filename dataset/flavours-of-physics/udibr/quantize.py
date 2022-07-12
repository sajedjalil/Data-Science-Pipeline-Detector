import pandas as pd
import numpy as np
from matplotlib import pylab as plt

fld = 'IPSig'
train = pd.read_csv("../input/training.csv")

X = np.sort(train[train.signal == 1][fld].values)
X = np.unique(X[1:]-X[:-1])
X = np.sort(X[1:]-X[:-1])
base = np.mean(X[(X>2e-16) & (X<2.3e-16)])/4.

M = 7
train['extra_'+fld] = ((train[fld].values)/base).astype(int) & M

signal = train[train.signal == 1]
background = train[train.signal == 0]


signal['extra_'+fld].hist(bins=range(0,M+1),label='signal', normed=True)
background['extra_'+fld].hist(bins=range(0,M+1),label='background',alpha=0.5, normed=True)
plt.title('quantize %s by %g'%(fld, base))
plt.legend()
plt.gcf().savefig('graph.png')

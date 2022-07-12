import pandas as pd
import numpy as np
from matplotlib import pylab as plt

fld = 'IPSig'
train = pd.read_csv("../input/training.csv")
test = pd.read_csv("../input/test.csv")

X = np.sort(train[train.signal == 1][fld].values)
X = np.unique(X[1:]-X[:-1])
X = np.sort(X[1:]-X[:-1])
base = np.mean(X[(X>2e-16) & (X<2.3e-16)])/4.
print(base)
M = 7
train['extra_'+fld] = ((train[fld].values)/base).astype(int) & M
test['extra_'+fld] = ((test[fld].values)/base).astype(int) & M

train = train[["id",'extra_'+fld]]
test = test[["id",'extra_'+fld]]

train.to_csv("train_extra.csv",index=False)
test.to_csv("test_extra.csv",index=False)
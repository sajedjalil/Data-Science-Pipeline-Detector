# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input/train_1"]).decode("utf8"))
# mat = loadmat('../input/train_1/1_1_0.mat')
# print(mat['dataStruct'].dtype.names)
# print(mat['dataStruct']['sequence'])
# names = mat['dataStruct'].dtype.names
# ndata = {n: mat['dataStruct'][n][0, 0] for n in names}

def mat_to_pandas(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    sequence = -1
    if 'sequence' in names:
        sequence = mat['dataStruct']['sequence']
    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0]), sequence

# interictal date
data,sequence = mat_to_pandas('../input/train_1/1_1_0.mat')
shape = data.shape
x = np.linspace(0, shape[0]/1000,shape[0])

# all signal in one picture
plt.figure()
for i in range(0,shape[1]):
    plt.plot(x, data[i+1].tolist())
plt.savefig('igg_1_1_1_0.png')

# mean of all signal
plt.figure()
sig = np.zeros((shape[0]),np.float)
for i in range(0,shape[1]):
    sig = sig+[abs(i) for i in data[i+1].tolist()]
plt.plot(x, sig)
plt.savefig('igg_1_1_1_0_means.png')

print("means and variance ")
print(np.mean(sig))
print(np.var(sig))

for i in range(0,shape[1]):
    plt.figure()
    plt.plot(x,data[i+1].tolist())
    plt.savefig('igg_1_1_1_0_'+str(i)+'.png')




# preictal date
data,sequence = mat_to_pandas('../input/train_1/1_1_1.mat')

shape = data.shape
x = np.linspace(0, shape[0]/1000,shape[0])

# all signal in one picture
plt.figure()
for i in range(0,shape[1]):
    plt.plot(x, data[i+1].tolist())
plt.savefig('igg_1_1_1_1.png')

# mean of all signal
plt.figure()
sig = np.zeros((shape[0]),np.float)
for i in range(0,shape[1]):
    sig = sig+[abs(i) for i in data[i+1].tolist()]
plt.plot(x, sig)
plt.savefig('igg_1_1_1_1_means.png')

print("means and variance ")
print(np.mean(sig))
print(np.var(sig))

for i in range(0,shape[1]):
    plt.figure()
    plt.plot(x,data[i+1].tolist())
    plt.savefig('igg_1_1_1_1_'+str(i)+'.png')













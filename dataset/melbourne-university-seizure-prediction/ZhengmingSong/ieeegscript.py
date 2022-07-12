# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pprint import pprint

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


import os
from scipy.io import loadmat

# Any results you write to the current directory are saved as output
labels = pd.read_csv('../input/train_and_test_data_labels_safe.csv')


# find out how many safe data clips are available for each patients
# for i in range(1,4):
#     safe_labels =labels.loc[(labels.safe == 1) & (labels.image.str.contains('{id}_[0-9]+_'.format(id = i)))]
#     print(i,':',len(safe_labels))

'''
output:
1 : 720
2 : 1986
3 : 2058
'''

# choose 2nd patients to start
safe_labels2 =labels.loc[(labels.safe == 1) & (labels.image.str.contains('{id}_[0-9]+_'.format(id = 2)))]

def mat_to_data(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return ndata

path = '../input/train_2/'

print(safe_labels2['image'])
label = safe_labels2['class'].iloc[0]
mat_path = path + safe_labels2['image'].iloc[0]
mat = loadmat(mat_path)
names = mat['dataStruct'].dtype.names
ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
data = ndata['data'][0]
print(label)
print(data)

for label in safe_labels2['image']:
    mat_path = path + label
    mat = loadmat(mat_path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    print(ndata['sequence'])
    
    



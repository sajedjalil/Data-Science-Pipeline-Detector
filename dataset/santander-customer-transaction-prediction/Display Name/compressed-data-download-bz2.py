# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import bz2

for fname in ['train.csv', 'test.csv', 'sample_submission.csv']:
    ifname = '../input/' + fname
    print(ifname)
    
    data = None
    with open(ifname, 'rb') as ifile:
        data = ifile.read()
    print('Data read, {} bytes'.format(len(data)))

    ofname = fname + '.bz2'
    print('Saving {}'.format(ofname))
    with bz2.open(ofname, mode='w') as ofile:
        ofile.write(data)
    wb = os.path.getsize(ofname)
    print('Data saved and compressed, {} bytes, {:.1f}%'.format(wb, wb / len(data) * 100))

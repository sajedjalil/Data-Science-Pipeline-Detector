# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import numpy as np
from scipy.spatial.distance import cdist

trainfile = '/kaggle/input/liverpool-ion-switching/train.csv'
testfile = '/kaggle/input/liverpool-ion-switching/test.csv'

X = np.genfromtxt(trainfile, delimiter=',', skip_header=1)
# time,signal,open_channels
time = X[:, 0]
signal = X[:, 1].reshape(-1, 1)
target_channels = X[:, 2]

# calculate medians for signal based on the nr. of open channels
medians = []

for channel in np.unique(target_channels):
    medians.append(np.median(signal[target_channels == channel]))

medians = np.array(medians)

test = np.genfromtxt(testfile, delimiter=',', skip_header=1)
signal = test[:, 1]

# calculate distances
distances = cdist(signal.reshape(-1, 1), medians.reshape(-1, 1))
predictions = np.argsort(distances, axis=1)

# for stacking, we need a 2D array
predictions = predictions[:, 0:1]
submission = np.hstack((test[:, 0:1], predictions))
print('Submission shape:', submission.shape)
np.savetxt('submission.csv', submission,  header='time,open_channels', fmt=['%.4f', '%d'], delimiter=',', comments='')
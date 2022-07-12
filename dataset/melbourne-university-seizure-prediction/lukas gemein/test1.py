# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def log_energy(data):
    return math.log(np.std(data))

file_pattern='../input/train_1/1_1_0.mat'
num_splits=60
func=log_energy

files = sorted(glob.glob(file_pattern), key=natural_key)
n_files = len(files)
feature = np.zeros((n_files*num_splits,16))
for i in range(n_files):
    path = files[i]
    try:
        mat = loadmat(path)
        data = mat['dataStruct']['data'][0, 0]
        print (data)
        exit(1)
        split_length = data.shape[0]/num_splits
        for s in range(num_splits):
            split_start = split_length*s
            split_end = split_start+split_length
            for c in range(16):
                channel_data = data[split_start:split_end,c]
                zero_fraction = float(channel_data.size - np.count_nonzero(channel_data))/channel_data.size
                # Exclude sections with more than 10% dropout
                if zero_fraction > 0.1:
                    feature[i*num_splits+s,c] = float('nan')
                else:
                    feature[i*num_splits+s,c] = func(channel_data)
    except:
        for s in range(num_splits):
            for c in range(16):
                feature[i*num_splits+s,c] = float('nan')
print (feature)




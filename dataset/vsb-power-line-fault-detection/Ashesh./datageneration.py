import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pyarrow.parquet as pq
import seaborn as sns
import scipy.fftpack as fftpack


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

meta_train_df = pd.read_csv('../input/metadata_train.csv').set_index('signal_id')
meta_test_df = pd.read_csv('../input/metadata_test.csv').set_index('signal_id')

def get_raw_train_df(cols):
    data = pq.read_pandas('../input/train.parquet', columns=cols)
    output = data.to_pandas()
    del data
    return output

def get_raw_test_df(cols):
    return pq.read_pandas('../input/test.parquet', columns=cols).to_pandas()
    
n_freq = 10000

def pick_freq(series):
    return fftpack.rfft(series)[:n_freq]

def get_freq_df(train_test_switch):
    """
    Convert to freq domain and return top 10K frequencies.
    """
    meta_df = None
    if train_test_switch == 'train':
        meta_df = meta_train_df
        get_raw_df = get_raw_train_df
    elif train_test_switch == 'test':
        meta_df = meta_test_df
        get_raw_df = get_raw_test_df
    
    raw_df = None

    i = 0
    columns =list(map(str,meta_df.index.tolist()))
    concur = 50
    j = 0
    for i in range(0, len(columns), concur) :
        cols =columns[i:(i+concur)]
        j+= 1
        if j %10 == 0:
            print(np.round(i/len(columns) * 100), '% Complete')
        data = get_raw_df(cols).apply(pick_freq)
        if raw_df is None:
            raw_df = data
        else:
            raw_df[cols] = data
        del data
    return raw_df
    
raw_train_df = get_freq_df('train')
raw_train_df.to_csv('train_freq.csv')
del raw_train_df


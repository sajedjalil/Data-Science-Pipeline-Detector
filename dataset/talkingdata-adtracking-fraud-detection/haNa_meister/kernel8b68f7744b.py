# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
def readCSV():
    train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
    test_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'uint32'
    }

    # Read the last lines because they are more impacting in training than the starting lines
    frm = 184903890
    nchunk = 40000000
    frm = frm - nchunk
    data = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/train.csv',
                       skiprows=range(1, frm), nrows=nchunk, usecols=train_columns,
                       dtype=dtypes)
    data.info()
    data.to_csv('train.csv')
pass

readCSV()
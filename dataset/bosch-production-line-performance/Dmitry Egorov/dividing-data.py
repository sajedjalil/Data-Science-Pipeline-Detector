# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


from io import StringIO as sio
from random import random

num_path = '../input/train_numeric.csv'
date_path = '../input/train_date.csv'

num_buf = sio()
date_buf = sio()

frac = 0.005

with open(num_path, 'r') as numfile, open(date_path, 'r') as datefile:
    # headers
    num_buf.write(numfile.readline())
    date_buf.write(datefile.readline())
    
    nlines = 0
    for numl, datel in zip(numfile, datefile):
        if random() < frac:
            num_buf.write(numl)
            date_buf.write(datel)
        
        nlines += 1
        if nlines % 100000 == 0: print("progress: {0}".format(nlines))
        

num_buf.seek(0)
date_buf.seek(0)
num_df = pd.read_csv(num_buf)
date_df = pd.read_csv(date_buf)

print(num_df.shape)
print(date_df.shape)

df = pd.concat([num_df, date_df.drop(['Id'], axis=1)], axis=1)

print(df.shape)
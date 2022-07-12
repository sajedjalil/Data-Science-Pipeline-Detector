# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

from random import random
from io import StringIO as sio

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

date_path = '../input/train_date.csv'
num_path = '../input/train_numeric.csv'



num_buf = sio()
date_buf = sio()

with open(num_path, 'r') as numfile, open(date_path, 'r') as datefile:
    numhead = numfile.readline()
    datehead = datefile.readline()
    
    num_buf.write(numhead)
    date_buf.write(datehead)
    
    nlines = 0
    for numln, dateln in zip(numfile, datefile):
    
        if random() < 0.0005:
            num_buf.write(numln)
            date_buf.write(dateln)
            
        nlines += 1
        if nlines % 100000 == 0: print("progress: {0}".format(nlines))
        

num_buf.seek(0)
date_buf.seek(0)


num_df = pd.read_csv(num_buf)
date_df = pd.read_csv(date_buf)

print(len(num_df.index))
print(num_df.shape)
print()
print(len(date_df.index))
print(date_df.shape)








    
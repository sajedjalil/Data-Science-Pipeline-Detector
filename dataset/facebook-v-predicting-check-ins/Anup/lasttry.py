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
import pandas as pd
import numpy as np
import datetime
import time
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

#train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('../input/test.csv')
#start_time = time.time()

#print (test.shape)

size = 10.0;

x_step = 0.2
y_step = 0.2

x_ranges = zip(np.arange(0, size, x_step), np.arange(x_step, size + x_step, x_step))
y_ranges = zip(np.arange(0, size, y_step), np.arange(y_step, size + y_step, y_step))

for x,y in x_ranges:
    print ("hel")
    print (x)
    print (y)
    for a,b in y_ranges:
        print (a)
        print (b)

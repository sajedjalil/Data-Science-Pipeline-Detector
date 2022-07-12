# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

for f in os.listdir('../input'):
    print(f.ljust(30)+str(round(os.path.getsize('../input/'+f)/1000000,2))+'MB')

df_train=pd.read_csv('../input/train.csv',nrows=50000)
print(df_train)
df_test = pd.read_csv('../input/test.csv',nrows=50000)
print(df_test)



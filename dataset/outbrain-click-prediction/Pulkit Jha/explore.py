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
import os
pd.set_option('display.max_columns',None)
for x in os.listdir('../input/'):
    print(x.ljust(40),round(os.path.getsize('../input/' + x)/1000000,2),'MB')
    
    
train = pd.read_csv('../input/clicks_train.csv', header=0)
print(train.head())
print(train.shape)

test = pd.read_csv('../input/clicks_test.csv', header=0)
print(test.head())
print(test.shape)

print(train.groupby('display_id')['ad_id'].count().value_counts())
print(test.groupby('display_id')['ad_id'].count().value_counts())
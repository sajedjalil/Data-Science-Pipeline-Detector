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
os.system("wc -l ../input/train_ver2.csv")
os.system("wc -l ../input/test_ver2.csv")

train = pd.read_csv('../input/train_ver2.csv', nrows = 100000)
test  = pd.read_csv('../input/test_ver2.csv', nrows = 100000)

for x in train.columns:
    print(x + " : " + str(train[x].nunique()))

pd.set_option('display.max_columns',None)
#print(train.head())
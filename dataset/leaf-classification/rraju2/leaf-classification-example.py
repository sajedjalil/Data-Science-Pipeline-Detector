# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


batch_size = 100
features = 192

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

le = LabelEncoder().fit(train.species) 
y_train = le.transform(train.species)
x_train = train.drop(train.columns[[0,1]],axis=1).values
x_test = test.drop(test.columns[[0]],axis=1).values
print(x_train)
print(y_train)



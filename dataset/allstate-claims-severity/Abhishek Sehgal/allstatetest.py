# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("../input/train.csv")

dfSize = df.shape

nFeat = dfSize[1]
cols = list(df.columns.values)
encoder = LabelEncoder()

catData = encoder.fit_transform(df[cols[1]])

for i in range(2,117):
	temp =  encoder.fit_transform(df[cols[i]])
	catData = np.vstack((catData, temp))

data = np.hstack((catData.T, df[cols[117:nFeat-1]].as_matrix()))
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 19:20:31 2018

@author: ahmed
"""
import numpy as np
import pandas as pd

dataset = pd.read_csv('../input/train.csv')

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_nor = sc_X.fit_transform(X)
sc_y = StandardScaler()
y_nor = sc_y.fit_transform(y.reshape(-1,1))
y_nor = y_nor.reshape(len(y_nor),)


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_nor, y_nor, test_size = 0.2, random_state = 0)
############################################################

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

train_score = regressor.score(X_train, y_train)
test_score = regressor.score(X_valid, y_valid)

print('train_score' , train_score)
print('test_score' , test_score)
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
#  -*- coding: utf-8 -*-
"""
Created on Tian Feng  8 22:09:44 2019

@author: Tian Feng
"""

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  # Gaussian naive Bayes classifier
from sklearn.preprocessing import LabelEncoder

print("\n\nTop of the training data:")
print(train.head(2))
print("\n\nTop of the test data:")
print(test.head(2))

print(train.Parch.value_counts())

print("\n\nTraining set survival:")
print(train.Survived.value_counts())

# 读取训练集和数据集
f1=open("../input/train.csv","rb")
train_data = np.loadtxt(f1,delimiter=',',skiprows=1)
f1.close()
train_x = train_data[:,1:54]
train_y = train_data[:,55]

f2 = open("../input/test.csv","rb")
test_data = np.loadtxt(f2,delimiter=',',skiprows=1)
f2.close()
test_x = test_data[:,1:54]


NBayes = GaussianNB()
NBayes.fit(train_x, train_y)
pre_test = NBayes.predict(test_x) # 进行预测
real_test_data = conpute_euclid_distance(pd.read_csv("../input/test.csv"))

result = pd.DataFrame({"Id": real_test_data.iloc[:,0].values, "Cover_Type": y_predict}, columns=['Id', 'Cover_Type'])
result.to_csv("submission.csv", index=False)
testResults.to_csv('titanic_prediction.csv', index=False)

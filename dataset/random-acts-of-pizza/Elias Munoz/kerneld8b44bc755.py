# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pd_train = pd.read_json('../input/train.json', orient='columns')
pd_test = pd.read_json('../input/test.json', orient='columns')

np_test = np.array(pd_test)
np_train = np.array(pd_train)

print(np_train.shape)


# X = np_train[:,0] # giver_username_if_known # slice a column
X = np_train[:,6] # request_text # text datas # Anyway click right side, Draft Environment, train.json.
Y = np_train[:,22] # requester_received_pizza # True or False
# ---------------- 
# Validation
# Hold-out, not K-hold
shuffle = np.random.permutation(np.arange(X.shape[0])) # generate randomized numbers as many as rows # 0 to 4039

# print(shuffle)
# max(shuffle) 4039
X, Y = X[shuffle], Y[shuffle] # 0 to 4039 # randamyzed array contents

# same data
print('data shape: ', X.shape) # 4040 rows
print('label shape:', Y.shape) # 4040 rows

# print(X) # text datas
# print(Y) # True or False


#l=len(X)
l = int(4040)

# closs validation, 
# l is not number 1, just alfabet L, l = 4040
# data -> Text datas
# labels -> True or False
train_data, train_labels = X[:int(l/2)], Y[:int(l/2)] # Slice rows to half 

# print(X) # text datas
# print(len(X)) # 4040
# print(l) # 4040
# train_data.shape # 2020
print(int((3*l)/4)) # 3030

# 2020 to 3030, slice 1010 from half line
# data -> Text datas
# labels -> True or False
dev_data, dev_labels = X[int(l/2):int((3*l)/4)], Y[int(l/2):int((3*l)/4)]

dev_data.shape, dev_labels.shape # 1010, 1010


# 3030 to 4040, slice 1010 from 3030
# data -> Text datas
# labels -> True or False
test_data, test_labels = X[int((3*l)/4):], Y[int((3*l)/4):] # 3030 to 4040, Slice 1010

# Any results you write to the current directory are saved as output.



vect = CountVectorizer() # Bag of Words

type(vect) # sklearn.feature_extraction.text.CountVectorizer

# toarray() changes to numpy object from CountVectorizer
data = vect.fit_transform(train_data).toarray() 
devdata = vect.transform(dev_data).toarray()
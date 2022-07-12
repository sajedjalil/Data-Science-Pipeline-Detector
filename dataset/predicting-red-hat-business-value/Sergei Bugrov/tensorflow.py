import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder

LEARNING_RATE = 1e-2
BATCH_SIZE = 256

train_X = pd.read_csv('../input/act_train.csv')
people = pd.read_csv('../input/people.csv')
test_X = pd.read_csv('../input/act_test.csv')

train_X = pd.merge(train_X, people, on='people_id')
train_X = train_X.fillna(value='N')
train_y = train_X['activity_category']
train_X.drop(['activity_category', 'people_id', 'activity_id'], axis=1, inplace=True)

test_X = pd.merge(test_X, people, on='people_id')
test_X = test_X.fillna(value='N')
test_y = test_X['activity_category']
test_X.drop(['activity_category', 'people_id', 'activity_id'], axis=1, inplace=True)

print(train_X.head())
print(test_X.head())
print(train_y.head())
print(test_y.head())

enc = OneHotEncoder()

train_X = enc.transform(train_X).toarray()

print(test_X)
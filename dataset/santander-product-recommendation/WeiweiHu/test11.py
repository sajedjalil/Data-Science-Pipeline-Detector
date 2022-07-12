"""
Code based on BreakfastPirate Forum post
__author__ : SRK
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns

#matplotlib inline

color = sns.color_palette()

data_path = "../input/"
train_file = data_path + "train_ver2.csv"
test_file = data_path + "test_ver2.csv"


train = pd.read_csv(data_path+train_file, usecols=['ncodpers'])
test = pd.read_csv(data_path+test_file, usecols=['ncodpers'])
print("Number of rows in train : ", train.shape[0])
print("Number of rows in test : ", test.shape[0])


train_unique_customers = set(train.ncodpers.unique())
test_unique_customers = set(test.ncodpers.unique())
print("Number of customers in train : ", len(train_unique_customers))
print("Number of customers in test : ", len(test_unique_customers))
print("Number of common customers : ", len(train_unique_customers.intersection(test_unique_customers)))












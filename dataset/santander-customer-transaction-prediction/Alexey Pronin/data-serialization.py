#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:46:03 2019

@author: alexey
"""
########################################################################################################
#
# REDUCE THE MEMORY USAGE AND READ THE COMPETION DATA FASTER!!!
#
########################################################################################################
# The code below does the following:
#    1. Assigns data types to the columns of train and test to reduce memory usage (50% reduction).
#    2. Separates 'target' and 'ID_code' columns from the training and testing data.
#    3. Serializes the data saving them into the feather file format. Reading feather files from the disk
#       happens much faster than reading CSV files, so you can use the output of this script every time 
#       when you need to read the data instead of reading the CSV's.
#    4. For convenience, the column names of test and train are saved as well as pickle files.
########################################################################################################

###################################################################
# Import libraries
###################################################################

import feather
import pickle
import pandas as pd
import numpy as np

###################################################################
# Specifying paths
###################################################################

on_kaggle = 1 # 1 if running on Kaggle; 0 if running locally

path = '../input/'

if on_kaggle:
    path = ''

path_train = path + 'train.feather'
path_test = path + 'test.feather'

path_train_colnames = path + 'train_colnames.pickle'
path_test_colnames = path + 'test_colnames.pickle'

path_target = path + 'target.feather'

path_train_ids = path + 'train_ids.feather'
path_test_ids = path + 'test_ids.feather'


###################################################################
# Making a dictionary holding the data types (specifying the data
# types will help us to reduce the memory usage by about 50%):
###################################################################

# Generating the list of names for the numeric features:

col_names = ['var_' + str(i) for i in range(200)]

# Unfortunately, feather does not support the np.float16
#  data type, so we have to use np.float32 for numeric 
# features 

col_types = [np.float32 for i in range(200)]

types = dict(zip(col_names, col_types))

types['target'] = np.uint8
types['ID_code'] = str

"""
# Uncomment this part and run it if you want to see  
# what the memory usage is without specifying the data types
###################################################################
# Baseline (without specifying the data types)
###################################################################

print("Reading train data...")

train = pd.read_csv('../input/train.csv')

print("Reading test data...")

test = pd.read_csv('../input/test.csv')

print("Memory usage train: {0:.2f}MB".
      format(train.memory_usage().sum() / (1024**2)))

print("Memory usage test: {0:.2f}MB".
      format(train.memory_usage().sum() / (1024**2)))
"""
###################################################################
# Reading and sorting the data:
###################################################################

print("Reading train data...")

train = pd.read_csv('../input/train.csv', dtype=types)

print("Reading test data...")

test = pd.read_csv('../input/test.csv', dtype=types)

# Separating the 'target' and 'ID_code' columns from
# the numeric features:

target = train.pop('target')
train_ids = train.pop('ID_code')
test_ids = test.pop('ID_code')

###################################################################
# Checking the data types and memory usage:
###################################################################

print("The data types in the train: ")
train.dtypes.value_counts()

print("Memory usage for the train: {0:.2f}MB".
      format(train.memory_usage().sum() / (1024**2)))

print("The data types in the train_ids: ")
train_ids.dtypes

print("Memory usage for train_ids: {0:.2f}MB".
      format(train_ids.memory_usage() / (1024**2)))

print("The data types in the target: ")
target.dtypes

print("Memory usage for the target: {0:.2f}MB".
      format(target.memory_usage() / (1024**2)))

print("The data types in the test: ")
test.dtypes.value_counts()

print("Memory usage the test: {0:.2f}MB".
      format(test.memory_usage().sum() / (1024**2)))

print("The data types in the test_ids: ")
test_ids.dtypes

print("Memory usage for the test_ids: {0:.2f}MB".
      format(test_ids.memory_usage() / (1024**2)))
      
###################################################################
# Saving the data to feather and pickle files:
###################################################################

print("Saving train to a feather files...")

feather.write_dataframe(train, path_train)

print("Saving the column names for train into a pickle file...")

pickling_on = open(path_train_colnames,"wb")
pickle.dump(train.columns, pickling_on)
pickling_on.close()

print("Saving target to a feather files...")

pd.DataFrame({'target' : target.values}).to_feather(path_target)

print("Saving train_ids to a feather files...")

pd.DataFrame({'ID_code' : train_ids.values}).to_feather(path_train_ids)

print("Saving test to a feather files...")

feather.write_dataframe(test, path_test)

print("Saving the column names for test into a pickle file...")

pickling_on = open(path_test_colnames,"wb")
pickle.dump(test.columns, pickling_on)
pickling_on.close()

print("Saving test_ids to a feather files...")

pd.DataFrame({'ID_code' : test_ids.values}).to_feather(path_test_ids)

###################################################################

print("All done!")

###################################################################
# -*- coding: utf-8 -*-
#Please ignore the error message.
# This script uses sklearn_pandas which isn't installed at kaggle
# pip install sklearn-pandas
import pandas as pd

from sklearn import preprocessing

from sklearn_pandas import DataFrameMapper

# We'll use the pandas library to read CSV files into dataframes


def loadFiles(trainPath, testPath):
    # Read competition data files
    train = pd.read_csv(trainPath, index_col=0)
    test  = pd.read_csv(testPath, index_col=0 )
   
    labels = train.Hazard
    train.drop('Hazard', axis=1, inplace=True)
    
    test_ind = test.index
    
    #concatanate the pandas dataframes
    temp = pd.concat([train,test])

    #inspired from http://stackoverflow.com/questions/24745879/
    binaries = ['T1_V6','T1_V17','T2_V3','T2_V11','T2_V12']
    encoders = ['T1_V4','T1_V5','T1_V7','T1_V8','T1_V9','T1_V11','T1_V12','T1_V15','T1_V16','T2_V5','T2_V13']
    scalars = ['T1_V1','T1_V2','T1_V3','T1_V10','T1_V13','T1_V14','T2_V1','T2_V2','T2_V4','T2_V6','T2_V7','T2_V8','T2_V9','T2_V10','T2_V14','T2_V15']
    
    mapper = DataFrameMapper(
        [(binary, preprocessing.LabelBinarizer()) for binary in binaries] +
        [(encoder, preprocessing.LabelEncoder()) for encoder in encoders] +
        [(scalar, preprocessing.StandardScaler()) for scalar in scalars]
    )
    
    tempMapped = mapper.fit_transform(temp)

    #split them apart  again
    train_ = tempMapped[:len(train)]
    test_ = tempMapped[len(test):]
    
    return train_, test_, labels, test_ind
   
print("Loading Data....")
train, test, labels, test_ind = loadFiles("input/train.csv", "input/test.csv")
print(len(train), len(labels), len(test), len(test_ind))

#I like to have a test set to play with so the next lines generate a split of the
#provided training set into a train and test set (not to be confused with the actual competion
#test set)
from sklearn.cross_validation import train_test_split
data_train, data_test, labels_train, labels_test = train_test_split(train, labels, test_size=0.20, random_state=42)

   
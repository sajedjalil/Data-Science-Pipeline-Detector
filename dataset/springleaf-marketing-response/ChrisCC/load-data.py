# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np 
import scipy as sc
from sklearn import preprocessing, feature_extraction

# The competition datafiles are in the directory ../input
# List the files we have available to work with
print("> ls ../input")
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Read train data file:
trainData = pd.read_csv("../input/train.csv")
testData = pd.read_csv("../input/test.csv")


trainSize = trainData.shape[0]
testSize = testData.shape[0]
print ("Train: %d rows loaded" % (trainSize))
print ("Test: %d rows loaded" % (testSize))

# fullData=pd.concat([trainData,testData])
# del trainData; del testData

# #Define columns: categorical, numberic and time serial
# labelCol =['target']
# idCol =['ID']
# catCols = list(trainData.dtypes[trainData.dtypes=='object'].index)
# numCols = list(trainData.dtypes[trainData.dtypes=='int64'].index) + list(trainData.dtypes[trainData.dtypes=='float64'].index); numCols.remove('ID'); numCols.remove('target')

# for col in catCols:
#     fullData[col]=LBL.fit_transform(fullData[col])
#     print (col) 
    
# fullEncCat = pd.get_dummies(fullData[catCols],columns=catCols)
# fullEncNum = pd.get_dummies(fullData[numCols],columns=numCols)
# dummyCatCols = list(fullEncCat.columns)
# dummyNumCols = list(fullEncNum.columns)    

# fullData = pd.concat([fullData.reset_index(), fullEncCat.reset_index(), fullEncNum.reset_index()],axis=1)
# fullData.set_index('ID')
# fullData[:trainSize].to_csv('train_.csv')
# fullData[trainSize:].to_csv('test_.csv')


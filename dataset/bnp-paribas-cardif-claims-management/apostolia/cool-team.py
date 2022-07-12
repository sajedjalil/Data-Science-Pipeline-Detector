# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

from pandas import read_csv
import csv
import numpy
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import time
from sklearn import tree
from sklearn import linear_model


def StringToNumber(string):
    if string==string:  #catch NaN
        sum = 0
        for i in range(len(string)):
            sum += (ord(string[i].lower())-ord('a')+1)
        return sum
    else:
        return string
    
    return

def extractData(userID, predict):
    userTargetPredict = open("result.csv", "w")
    userTargetPredict.write("ID,PredictedProb")    
    userTargetPredict.write("\n")

    count = 0
    
    for i in predict:
    
        userTargetPredict.write(str(userID[count]))
        userTargetPredict.write(",")
        userTargetPredict.write(str(i))
        userTargetPredict.write("\n")
     
        count = count + 1    
        
    userTargetPredict.close() 
    return

listStringColumns = [3,22,24,30,31,47,52,56,66,71,74,75,79,91,107,110,112,113,125]

trainData = (read_csv("../input/train.csv").drop(['ID'],axis=1))
testData = read_csv("../input/test.csv")

trainTarget = trainData.target

testUsers = ['ID']
testUsers = testData.ID

testData=testData.drop(['ID'],axis=1)
trainData=trainData.drop(['target'],axis=1)

for i in listStringColumns:
    name = "v" + str(i)
    trainData[name] = trainData[name].apply(StringToNumber)
    testData[name] = testData[name].apply(StringToNumber)

imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(trainData)
imp.fit(testData)
trainData=imp.transform(trainData)
testData=imp.transform(testData)

print ("Local current time after modeling:", time.localtime(time.time()))


clf = OneVsRestClassifier(LinearSVC())
clf = clf.fit(trainData, trainTarget)

print ("Local current time after modeling:", time.localtime(time.time()))
predicted = clf.predict(testData)
extractData(testUsers, predicted)
print ("Local current time end:", time.localtime(time.time()))
# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

from pandas import read_csv
import csv
import numpy
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OutputCodeClassifier
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
#import time
from sklearn import linear_model

def StringToNumber(string):
    if string == string:
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
trainData = imp.transform(trainData)
testData = imp.transform(testData)

n = numpy.array(trainData)
trainStd =  n.std(0)
print (len(trainStd))
print (trainStd)



print('std (1)')
testing =  n.std(1)
print (len(testing))


for i in range(len(trainStd)):
    if trainStd[i]<= 0.5:
        print(i)
        
trainData = imp.transform(trainData)
testData = imp.transform(testData)



print(len(trainData))
print(len(testData))



#print ("Local current time after modeling:", time.localtime(time.time()))

#clf = linear_model.Lasso(alpha = 0.05) 
#clf = OutputCodeClassifier(LinearSVC(),code_size=2, random_state=0)
#clf = clf.fit(trainData, trainTarget)

#print ("Local current time after modeling:", time.localtime(time.time()))
#predicted = clf.predict(testData)
#extractData(testUsers, predicted)
#print ("Local current time end:", time.localtime(time.time()))
# Any results you write to the current directory are saved as output.
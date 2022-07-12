# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
# Import necessary libraries
import csv as csv
import numpy as np
from sklearn.linear_model import LinearRegression

csvFileObjectTrain = csv.reader(open("../input/train.csv", 'r'))
csvFileObjectTest = csv.reader(open("../input/test.csv", 'r'))
header = next(csvFileObjectTest)
header = next(csvFileObjectTrain)

dataTest = []
for row in csvFileObjectTest:
    dataTest.append(row)
    
dataTrain = []
for row in csvFileObjectTrain:
    dataTrain.append(row)

def organiseSequenceData(row):
    # put each number in the sequence as the separate example
    row = list(map(int, row.split(',')))
    X = []
    if(len(row) > 200):
        print(len(row))
    for element in row:
        # check if element is too big to include polynomies
        #if element > 10**100: #100
        #    max_power = 1
       
        X.append([element])
    Y = X[1:len(X)]
    Y = np.array(Y)    
    X = np.array(X)
    
    Xtrain = X[0:len(X) - 1, 0::]
    Xtest = X[len(X) - 1, 0::]
    return(Xtrain, Y, Xtest) 

predictionFile = open("LinearRegressionIntegerSequence.csv", "w", newline = "")
predictionFileObject = csv.writer(predictionFile)
predictionFileObject.writerow(["Id", "Last"])
    
for row in dataTest:
    ID = row[0]
    if(len(list(map(int, row[1].split(',')))) == 1):
        predictionFileObject.writerow([ID, row[1]])
    else:
        Xtrain, Y, Xtest = organiseSequenceData(row[1])
        model = LinearRegression()
        model = model.fit(Xtrain, Y)
        prediction = model.predict(Xtest)
        if prediction == float('Inf'):
            predictionFileObject.writerow([ID, 0])
        else:    
            predictionFileObject.writerow([ID, int(prediction[0])])
    
predictionFile.close() 
            

            
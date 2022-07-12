# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 18:30:36 2017

@author: Fernando Cruz
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import	linear_model
from sklearn import cross_validation
from random import shuffle

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

X_train = train.loc[:, 'margin1':]
y_train = train["species"]

X_test = test.loc[:,'margin1':]

#PASSAR ESTAS FUNÇOES A FRENTE

def svm_model():
    clf = svm.SVC()
    svm_model = clf.fit(X_train, y_train)
    print(svm_model)

    predict_test = clf.predict(X_test)

    print(test.shape)
    print(predict_test.shape)

def KKN():
    knn = KNeighborsClassifier()
    kkn_model = knn.fit(X_train, y_train)
    print(kkn_model)
    
    print("Valores previstos: ", knn.p)
    
def evaluate_basis():
    predictions = []
    for i in range(10):
        predict = Logistic_regression()
        predictions.append(predict)    
    dic = {}
    for pred in range(len(predictions)):
        predict = predictions[pred]
        for id in range(len(predict)):
            if (test.loc[id,"id"], predict[id]) in dic:
                dic[(test.loc[id,"id"], predict[id])]+=1
            else:
                dic[(test.loc[id,"id"], predict[id])]=1
    print(len(dic))
    
#IGNORAR CONTEUDO ACIMA


def validation(reps=10):
    logistic = linear_model.LogisticRegression(C=1)
    logistic = logistic.fit(X_train, y_train)
    predictions = logistic.predict(X_test)
    ids = list(test.loc[:,"id"])
    species = list(np.unique(y_train))
    predictions_counts = np.zeros([len(ids),len(species)])
    lin = 0
    for pred in predictions:
        col = species.index(pred)
        predictions_counts[lin,col] += 1/len(predictions)
        lin += 1
    for rep in range(reps):
        train_indexes = list(range(train.shape[0]))
        shuffle(train_indexes)
        X_train_2 = train.loc[train_indexes, 'margin1':]
        y_train_2 = train.loc[train_indexes,"species"]
        logistic = linear_model.LogisticRegression(C=1)
        logistic = logistic.fit(X_train_2, y_train_2)
        predictions = logistic.predict(X_test)
        lin = 0
        for pred in predictions:
            col = species.index(pred)
            predictions_counts[lin,col] += 1/len(predictions)
            lin += 1
    return predictions_counts, species, ids

def write():
    probs, species, ids = validation()
    file = open("test_logistic_1.csv", "w")
    file.write("id")
    for specie in species:
        file.write(","+str(specie))
    file.write("\n")
    for i in range(len(probs)):
        file.write(str(ids[i]))
        for j in probs[i]:
            file.write(","+str(j))
        file.write("\n")
    
    
if __name__=="__main__":
    #KKN()
    #Logistic_regression(X_train,y_train)
    write()
    #validation()
    
    
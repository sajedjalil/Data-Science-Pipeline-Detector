# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from math import log
import json
import operator
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from pandas import DataFrame
import copy
import math
import random

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

def get_data(path):
    f = open(file=path,encoding='utf-8')
    data = json.load(f)
    return data

def clear_account(lists):
    wokey = {}
    wokey = wokey.fromkeys(lists)
    word_1 = list(wokey.keys())
    for i in word_1:
       wokey[i] = lists.count(i)
    return wokey

def sort_1(wokey):
    wokey_1 = {}
    wokey_1 = sorted(wokey.items(), key=lambda d:d[1], reverse=True)
    wokey_1 = dict(wokey_1)
    return wokey_1

def top_20(final_dict,allnum,all_dict):
    dict_list = list(final_dict.keys())
    TF_dict = copy.deepcopy(final_dict)
    IDF_dict = copy.deepcopy(final_dict)
    TF_IDF_dict = copy.deepcopy(final_dict)
    for item in dict_list:
        TF_dict[item] = TF_dict[item]/allnum                                
        IDF_dict[item] = math.log2(39774/all_dict[item]+1)
        TF_IDF_dict[item] = TF_dict[item]*IDF_dict[item]
    TF_IDF_dict = sort_1(TF_IDF_dict)
    key_list = list(TF_IDF_dict.keys())
    return key_list[0:20]

def get_features_list(train_data,cuisine):
    type_list = []
    for datai in train_data:
        for ingredients in datai['ingredients']:
            if datai['cuisine'] == cuisine:
                type_list.append(ingredients)
    return type_list
    
def calcshan(dataSet):
    lenDataSet = len(dataSet)
    p = {}
    H = 0.0
    for data in dataSet:
        currentLabel = data[-1]
        if currentLabel not in p.keys():
            p[currentLabel] = 0
        p[currentLabel] += 1
    for key in p:
        px = float(p[key]) / float(lenDataSet)
        H -= px * log(px, 2)
    return H

def splitData(dataSet,axis,value):   
    subDataSet=[]
    for data in dataSet:
        subData=[]
        if data[axis]==value:
            subData=data[:axis]  
            subData.extend(data[axis+1:]) 
            subDataSet.append(subData) 
    return subDataSet

def C45_chooseBestFeatureToSplit(dataset):
    numFeatures=len(dataset[0])-1
    baseEnt=calcshan(dataset)
    bestInfoGain_ratio=0.0
    bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i]for example in dataset]
        uniqueVals=set(featList) 
        newEnt=0.0
        IV=0.0
        for value in uniqueVals:   
            subdataset=splitData(dataset,i,value)
            p=len(subdataset)/float(len(dataset))
            newEnt+=p*calcshan(subdataset)
            IV=IV-p*log(p,2)
        infoGain=baseEnt-newEnt
        if (IV == 0):
            continue
        infoGain_ratio = infoGain / IV                 
        if (infoGain_ratio >bestInfoGain_ratio):        
            bestInfoGain_ratio = infoGain_ratio
            bestFeature = i 
    return bestFeature

def majorityCnt(classList): 
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1   
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) 
    return sortedClassCount[0][0]

def C45_createTree(dataset,labels):
    classList=[example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    bestFeat = C45_chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    C45Tree = {bestFeatLabel:{}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        C45Tree[bestFeatLabel][value] = C45_createTree(splitData(dataset, bestFeat, value), subLabels)
    return C45Tree

def classify(tree,label,testVec,c_set):
    firstFeat=list(tree.keys())[0]
    secondDict=tree[firstFeat]
    labelIndex=label.index(firstFeat)
    classLabel = str(random.sample(c_set,1))
    classLabel = classLabel.replace('[', '')
    classLabel = classLabel.replace(']', '')
    classLabel = classLabel.replace('\'', '')
    for key in secondDict.keys(): 
        if testVec[labelIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],label,testVec,c_set)
            else:
                classLabel=secondDict[key]
    return classLabel

if __name__ == '__main__':
    cuisine_set = set()
    recipe_set = set()
    features = set()
    recipe_list = list()
    all_ingredients = list()
    cuisine_list = list()

    train_path = '../input/train.json'
    train_data = get_data(train_path)
    N = len(train_data)
    for datai in train_data:
        cuisine_list.append(datai['cuisine'])  #39774list
        cuisine_set.add(datai['cuisine'])  # 20set
        recipe_list.append(datai['ingredients'])  # list39774
        for ingredients in datai['ingredients']:
            recipe_set.add(ingredients)  # 6714set
            all_ingredients.append(ingredients)#428275list

    lists = []
    for cuisine in cuisine_set:
        lists.append(get_features_list(train_data=train_data,cuisine=cuisine))

    recipe_clear_dict = clear_account(all_ingredients)
    recipe_final_dict = sort_1(recipe_clear_dict)
    top10 = list(recipe_final_dict.keys())[0:10]

    for i in range(len(lists)):
        dicti = clear_account(lists[i])
        final_dict = sort_1(dicti)
        listi = top_20(final_dict=final_dict,allnum=len(lists[i]),all_dict=recipe_final_dict)
        for j in listi:
            features.add(j)
            
    top10 = set(top10)
    features = features - top10
    M = len(features)
    features = list(features)

    data = np.zeros([N,M])
    for i in range(N):
        datai = train_data[i]
        for ingredients in datai['ingredients']:
            if ingredients in features:
                column = features.index(ingredients)
                data[i,column] = 1

    traindataset = DataFrame(data=data,columns=features)
    traindataset['cuisine'] = cuisine_list
    traindataset.to_csv('dataset.csv')

    test_path = '../input/test.json'
    test_data = get_data(test_path)
    testN = len(test_data)

    data = np.zeros([testN,M])
    for i in range(testN):
        datai = test_data[i]
        for ingredients in datai['ingredients']:
            if ingredients in features:
                column = features.index(ingredients)
                data[i,column] = 1
                
    testdataset = DataFrame(data=data,columns=features)
    testdataset.to_csv('testdataset.csv')
    
    df = pd.read_csv('dataset.csv')
    datalists = []
    for index in df.index:
        i = df.loc[index][1:]
        i = list(i)
        datalists.append(i)
    t1 = time.clock()
    tree = C45_createTree(dataset=datalists,labels=features)
    t2 = time.clock()
    print(t2-t1)
    
    testdf = pd.read_csv('testdataset.csv')
    features = list(testdf.columns.values)[1:]
    testlists = []
    t3 = time.clock()
    for index in testdf.index:
        j = testdf.loc[index][1:]
        j = list(j)
        testlists.append(classify(tree=tree,label=features,testVec=j,c_set=cuisine_set))
    t4 = time.clock()
    result = DataFrame()
    id_list = []
    for data in test_data:
        id_list.append(data['id'])
    result['id'] = id_list
    result['cuisine'] = testlists
    result.to_csv('submission.csv',index=False)
    print(t4-t3)
    print(result)
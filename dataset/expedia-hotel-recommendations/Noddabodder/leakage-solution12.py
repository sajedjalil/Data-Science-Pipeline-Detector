# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict
import numpy as np

def to_integer(dt_time):
    return 10000*dt_time.year + 100*dt_time.month + dt_time.day


print('Preparing arrays...')
f = open("../input/train.csv", "r")
f.readline()
total = 0
data = []

# Calc counts
while 1:
    line = f.readline().strip()
    total += 1

    if total % 1000000== 0:
        break
    
    arr = line.split(",")
    for x in range(len(arr)):
        if (len(arr[x])) == 0:
            arr[x] = '0'

    arr[0] = to_integer(datetime.datetime.strptime(arr[0], '%Y-%m-%d %H:%M:%S'))
    
    if arr[11]  !='0':
        arr[11] = to_integer(datetime.datetime.strptime(arr[11], '%Y-%m-%d'))
    
    if arr[12] !='0':
        arr[12] = to_integer(datetime.datetime.strptime(arr[12], '%Y-%m-%d'))
    
    for x in range(len(arr)):
        arr[x] = float(arr[x])
    
    data.append(arr)
    
data = np.array(data, dtype = np.int64)
predictors = data[:,0:23]

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
scores = cross_validation.cross_val_score(clf, predictors, data[:,23], cv=3)
print (scores)

#find records with no date
"""
arr = line.split(",")
if len(arr[12])<10:
    print (arr)
"""











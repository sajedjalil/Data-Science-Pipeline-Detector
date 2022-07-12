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

# Calc counts
while 1:
    line = f.readline().strip()
    total += 1

    if total % 10== 0:
        break

    arr = line.split(",")
    print (len(arr[6]))
    #print ((arr[6]is.null()))
    
    """
    arr[0] = to_integer(datetime.datetime.strptime(arr[0], '%Y-%m-%d %H:%M:%S'))
    arr[12] = to_integer(datetime.datetime.strptime(arr[12], '%Y-%m-%d'))
    arr[11] = to_integer(datetime.datetime.strptime(arr[11], '%Y-%m-%d'))
    #print (arr)
    
    for x in range(len(arr)):
        arr[x] = float(arr[x])
    
    print (arr)"""
    
    #data = np.array(arr[2:4], dtype = np.float64)
    #numpy.int32
    #numpy.float64

#print (type(data))
#print (data.shape)


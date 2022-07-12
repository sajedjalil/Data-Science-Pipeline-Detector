# -*- coding: utf-8 -*-
"""
Created on Thu Apr 06 22:09:54 2017

@author: amurp
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 06 21:48:33 2017

@author: amurp
"""

# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict
import ml_metrics as metrics
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import mixture
from sklearn.model_selection import cross_val_predict
import pandas as pd
import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict

def run_solution():
    print('Preparing arrays...')
    f = open("../input/train.csv", "r")
    f.readline()
    best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest1 = defaultdict(lambda: defaultdict(int))
    best_hotel_country = defaultdict(lambda: defaultdict(int))
    popular_hotel_cluster = defaultdict(int)
    total = 0

    # Calc counts
    while 1:
        line = f.readline().strip()
        total += 1

        if total % 10000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        book_year = int(arr[0][:4])
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        srch_destination_id = arr[16]
        is_booking = int(arr[18])   
        hotel_country = arr[21]
        hotel_market = arr[22]
        hotel_cluster = arr[23]
        
    print(arr)
        
    """predictors = [c for c in arr if c not in [arr[23]]]
    
    clf1 = SVC(gamma=0.5, C=3, probability=True)
    predictedCV1 = cross_val_predict(clf1, predictors, arr[23], cv=10)
    metrics.accuracy_score(arr[23], predictedCV1)"""
    
run_solution()
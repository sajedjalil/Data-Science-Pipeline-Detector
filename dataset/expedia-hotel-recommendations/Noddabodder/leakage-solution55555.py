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

def run_solution():
    print('Preparing arrays...')
    f = open("../input/train.csv", "r")
    f.readline()
    total = 0
    data = []
    break_count = 0

# Calc counts
    while 1:
        line = f.readline().strip()
        total += 1

        if total % 10000 == 0:
            print('Read {} lines...'.format(total))

#if line == '':
#   break
 
        if break_count == 10000:
             break

        arr = line.split(",")
        arr1 = np.array(arr)
        data.append(arr1)
        break_count += 1
    print(data)

run_solution()










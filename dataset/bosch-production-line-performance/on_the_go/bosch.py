

import pandas as pd
import numpy as np

from IPython.display import display, HTML
import time; start_time = time.time()
import math
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.externals import joblib
from scipy import sparse


# rdr = csv.reader(open("../input/train_date.csv"))
# line1 = next(rdr) 
# line2 = next(rdr)

trainCateDF = pd.read_csv('../input/train_categorical.csv', 
                      low_memory=False, #remove warning
                      encoding="ISO-8859-1", 
                      #usecols=[], 
                      #dtype={}, 
                      iterator=True, 
                      chunksize=10000) #change to memory capacity

trainDateDF = pd.read_csv('../input/train_date.csv', 
                       low_memory=False, #remove warning
                       encoding="ISO-8859-1", 
                       #usecols=[], 
                       #dtype={}, 
                       iterator=True, 
                       chunksize=10000) #change to memory capacity

trainNumDF = pd.read_csv('../input/train_numeric.csv', 
                       low_memory=False, #remove warning
                       encoding="ISO-8859-1", 
                       #usecols=[], 
                       #dtype={}, 
                       iterator=True, 
                       chunksize=10000) #change to memory capacity
                       
for df in trainCateDF:
    print('data in train_categorical.csv:\n')
    print(df.head(3))
    break
for df in trainDateDF:
    print('data in train_date.csv:\n')
    print(df.head(3))
    break
for df in trainNumDF:
    print('data in train_numeric.csv:\n')
    print(df.head(3))
    break


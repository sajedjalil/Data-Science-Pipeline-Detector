# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:09:21 2015

@author: Dipayan
"""


from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression


# A combination of Word lemmatization + LinearSVC model finally pushes the accuracy score past 80%


testdf = pd.read_json("../input/test.json") 
testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]) for lists in testdf['ingredients']]       
testdf['ingredients_string2'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip(' ') for lists in testdf['ingredients']]       

print (testdf['ingredients_string'][1:5])

print (testdf['ingredients_string2'][1:5])
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

traindf = pd.read_json("../input/train.json")
traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]  
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

testdf = pd.read_json("../input/test.json") 
testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       



corpustr = traindf['ingredients_string']
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
tfidftr=vectorizertr.fit_transform(corpustr).todense()
corpusts = testdf['ingredients_string']
vectorizerts = TfidfVectorizer(stop_words='english')
tfidfts=vectorizertr.transform(corpusts)

predictors_tr = tfidftr

targets_tr = traindf['cuisine']

predictors_ts = tfidfts


#classifier = LinearSVC(C=0.80, penalty="l2", dual=False)
parameters = {'C':[1, 10]}
#clf = LinearSVC()
clf = LogisticRegression()

classifier = grid_search.GridSearchCV(clf, parameters)

classifier=classifier.fit(predictors_tr,targets_tr)

predictions=classifier.predict(predictors_ts)
testdf['cuisine'] = predictions
testdf = testdf.sort('id' , ascending=True)

testdf[['id' , 'ingredients_clean_string' , 'cuisine' ]].to_csv("submission.csv")

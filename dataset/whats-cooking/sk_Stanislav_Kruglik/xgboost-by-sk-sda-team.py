from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, VotingClassifier ,BaggingClassifier ,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import codecs
import os
from sklearn.svm import SVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.neural_network import MLPClassifier



traindf = pd.read_json(codecs.open("../input/train.json",'r','utf-8') )
traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       
testdf = pd.read_json(codecs.open("../input/test.json",'r','utf-8') )
testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       

corpustr = traindf['ingredients_string']

estimators = [("tfidf", TfidfVectorizer(stop_words='english',
             ngram_range = ( 1 , 1 ),analyzer="word",
             max_df = .57 , binary=False ,max_features =6706, token_pattern=r'\w+' , sublinear_tf=False) ),
             ("hash", HashingVectorizer ( stop_words='english',
             ngram_range = ( 1 , 2 ),n_features  =6706,analyzer="word",token_pattern=r'\w+', binary =False))]


tfidftr = FeatureUnion(estimators).fit_transform(corpustr).todense()

corpusts = testdf['ingredients_string']

tfidfts = FeatureUnion(estimators).transform(corpusts)

predictors_tr = tfidftr

targets_tr = traindf['cuisine']

predictors_ts = tfidfts

classSVC = LinearSVC(C=0.3999, penalty="l2", dual=False) 

classSVC = classSVC.fit(predictors_tr,targets_tr)

predictions = classSVC.predict(predictors_ts) 

testdf['cuisine'] = predictions
testdf = testdf.sort('id' , ascending=True)

testdf[['id' , 'cuisine' ]].to_csv("final submission.csv", index=False)

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
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



# A combination of Word lemmatization + LinearSVC model finally pushes the accuracy score past 80%

traindf = pd.read_json("../input/train.json")

# strip()去除首尾空格
# join() 將字詞加入字串中
# 以,做區隔
traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]
# WordNetLemmatizer 從WordNet (英文詞彙庫) 中比對字詞，並將她還原
# re : Regular expression 用正規表示式來當作條件並執行函數動作
# re.sub : 將輸入字串中符合正規表示的規則取代成要取代的字串
# 將不是英文字母的字元取代成空格
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

#traindf.to_csv("traindf.csv", index=False)

testdf = pd.read_json("../input/test.json") 
testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       

#testdf.to_csv("testdf.csv", index=False)


corpustr = traindf['ingredients_string']
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
tfidftr = vectorizertr.fit_transform(corpustr).todense()
corpusts = testdf['ingredients_string']
vectorizerts = TfidfVectorizer(stop_words='english')
tfidfts = vectorizertr.transform(corpusts)


predictors_tr = tfidftr

targets_tr = traindf['cuisine']

predictors_ts = tfidfts


#classifier = LinearSVC(C=0.80, penalty="l2", dual=False)
parameters = {'C':[1, 10]}

lsvc = LinearSVC()
#lr = LogisticRegression()

lsvcClassifier = grid_search.GridSearchCV(lsvc, parameters)
#lsvcClassifier = classifier.fit(predictors_tr,targets_tr)

#lrClassifier = grid_search.GridSearchCV(lr, parameters)
#lrClassifier = classifier.fit(predictors_tr,targets_tr)


#clf =VotingClassifier(estimators=[('lsvc', lsvcClassifier), ('lr', lrClassifier)], voting='hard')

#predictions = clf.predict(predictors_ts)
predictions = lsvcClassifier.predict(predictors_ts)
testdf['cuisine'] = predictions
testdf = testdf.sort('id' , ascending=True)

#show the detail
#testdf[['id' , 'ingredients_clean_string' , 'cuisine' ]].to_csv("submission.csv")

#for submit, no index
testdf[['id' , 'cuisine' ]].to_csv("submission.csv", index=False)


# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:09:21 2015

@author: Dipayan
"""


from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re, sys
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


# A combination of Word lemmatization + LinearSVC model finally pushes the accuracy score past 80%
print('reading data...')
traindf = pd.read_json("../input/train.json")
traindf = traindf.iloc[:1000,:]
#traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]  
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

testdf = pd.read_json("../input/test.json") 
testdf = testdf.iloc[:1000,:]
#testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       



corpustr = traindf['ingredients_string']
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .6 , binary=False , token_pattern=r'\w+' , sublinear_tf=False, max_features = 1000)
tfidftr=vectorizertr.fit_transform(corpustr).todense()
corpusts = testdf['ingredients_string']
vectorizerts = TfidfVectorizer(stop_words='english')
tfidfts=vectorizertr.transform(corpusts).todense()

predictors_tr = tfidftr
print(type(predictors_tr))
targets_tr = traindf['cuisine']
#print(targets_tr)
predictors_ts = tfidfts
print(type(predictors_ts))
#sys.exit()
#classifier = LinearSVC(C=0.80, penalty="l2", dual=False)
#parameters = {'C':[1, 10]}
#clf = LinearSVC()
#clf = LogisticRegression()

print('training model...')
# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(predictors_tr,targets_tr)

print('predicting on test data...')
predictions = gbm.predict(predictors_ts)

#classifier = grid_search.GridSearchCV(clf, parameters)

#classifier=classifier.fit(predictors_tr,targets_tr)


# predictions=classifier.predict(predictors_ts)
# testdf['cuisine'] = predictions
# testdf = testdf.sort('id' , ascending=True)

# testdf[['id', 'cuisine' ]].to_csv("submission.csv",index=False)
print('writing submission file...')
submission = pd.DataFrame({ 'id': testdf['id'],
                            'cuisine': predictions })
submission.to_csv("submission.csv", index=False)

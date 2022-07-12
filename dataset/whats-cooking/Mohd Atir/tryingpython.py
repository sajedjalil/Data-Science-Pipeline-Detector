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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb


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

rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, criterion="entropy", random_state=11)

rf.fit(predictors_tr, targets_tr)



#classifier = LinearSVC(C=0.80, penalty="l2", dual=False)
parameters = {'C':[1, 10]}
#clf = LinearSVC()
clf = LogisticRegression()

classifier = grid_search.GridSearchCV(clf, parameters)

classifier=classifier.fit(predictors_tr,targets_tr)

predictions=classifier.predict(predictors_ts)
testdf['cuisine'] = predictions
testdf = testdf.sort('id' , ascending=True)

testdf[['id' , 'cuisine' ]].to_csv("submission.csv")


print("Train a XGBoost model")
params = {"objective": "multi:softmax",
          "num_class" : 20 ,
          "eta": 0.15,
          "max_depth": 7,
          "min_child_weight": 10,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=300
gbm = xgb.train(params, xgb.DMatrix(
    Matrix(data.matrix(predictors_tr)), 
    label = targets_tr), num_trees)

test_probs = (rf.predict_proba(predictors_ts) +
              gbm.predict(xgb.DMatrix(predictors_ts)))/2
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_xgboost_submission.csv", index=False)



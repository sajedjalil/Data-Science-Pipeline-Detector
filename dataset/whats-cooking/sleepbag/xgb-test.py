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
#from mlxtend.classifier import EnsembleClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import FeatureUnion
import json as js
import csv as csv
import scipy as scipy
import numpy as np
import pdb
from sklearn.neural_network import MLPClassifier


"""
traindf = pd.read_json(codecs.open("../input/train.json",'r','utf-8') )
print("read tran")

traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]

traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

#traindf.to_csv("traindf.csv", index=False,sep='\t', encoding='utf-8')

testdf = pd.read_json(codecs.open("../input/test.json",'r','utf-8') )
print("read test")
testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       

#testdf.to_csv("testdf.csv", index=False)

corpustr = traindf['ingredients_string']


estimators = [("tfidf", TfidfVectorizer(stop_words='english',
             ngram_range = ( 1 , 1 ),analyzer="word",
             max_df = .57 , binary=False ,max_features =6706, token_pattern=r'\w+' , sublinear_tf=False) ),
             ("hash", HashingVectorizer ( stop_words='english',
             ngram_range = ( 1 , 2 ),n_features  =6706,analyzer="word",token_pattern=r'\w+', binary =False))]


tfidftr = FeatureUnion(estimators).fit_transform(corpustr).todense()

print("tfidft tran")
corpusts = testdf['ingredients_string']
"""

######################## 
with open('../input/train.json') as json_data:
    data = js.load(json_data)
    json_data.close()
    
with open('../input/test.json') as json_data:
    test = js.load(json_data)
    json_data.close()

classes = [item['cuisine'] for item in data]
ingredients = [item['ingredients'] for item in data]
unique_ingredients = set(item for sublist in ingredients for item in sublist)
unique_cuisines = set(classes)

testIngredients = [item['ingredients'] for item in test]


big_data_matrix = scipy.sparse.dok_matrix((len(ingredients), len(unique_ingredients)), dtype=np.dtype(bool))

for d,dish in enumerate(ingredients):
    for i,ingredient in enumerate(unique_ingredients):
        if ingredient in dish:
            big_data_matrix[d,i] = True
print("train")
print(big_data_matrix)

clf2 = MLPClassifier (algorithm = 'sgd', alpha=0.001, hidden_layer_sizes=(100, 100, 100), random_state=1, activation='logistic' );
classMLP = clf2.fit(big_data_matrix, classes)

big_test_matrix = scipy.sparse.dok_matrix((len(testIngredients), len(unique_ingredients)), dtype=np.dtype(bool))
for d,dish in enumerate(testIngredients):
    for i,ingredient in enumerate(unique_ingredients):
        if ingredient in dish:
            big_test_matrix[d,i] = True
print("test")
print(big_test_matrix)
####################


#tfidfts = FeatureUnion(estimators).transform(corpusts)

print("tfidft test")


#predictors_tr = tfidftr

#targets_tr = traindf['cuisine']

#predictors_ts = tfidfts




#classSVC = LinearSVC(C=0.50, penalty="l2", dual=False)


#classSVC = classSVC.fit(predictors_tr,targets_tr)


print("predict")

#predictions = classSVC.predict(predictors_ts) 
#predictions = classMLP.predict() 

predictions = classMLP.predict(big_test_matrix) 

print(predictions)

print("predict finish")

#testdf['cuisine'] = predictions
#testdf = testdf.sort('id' , ascending=True)
print("sort by id")


#testdf[['id' , 'cuisine' ]].to_csv("submission.csv", index=False)

print("done")

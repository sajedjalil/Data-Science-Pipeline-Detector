#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:53:51 2018

@author: spotless
"""
import re
import json
import nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

with open('../input/whats-cooking-kernels-only/train.json') as json_file:
    json_data = json.load(json_file)

len(json_data)

# remove id for training data
list(map(lambda d: d.pop('id'), json_data));

# get the ingredients
json_ingredients = [x['ingredients'] for x in json_data]

# make lowercase, clear punctuation, and stem
# dashes are added so ngram isn't confused when lists are combined
lowercase = [[x.lower() for x in y] for y in json_ingredients]
punctuation_lowercase = [[re.sub(r'([^\s\w]|_)+', '', x) for x in y] for y in lowercase]
sno = nltk.stem.SnowballStemmer('english')
punctuation_lowercase_stemmed = [[sno.stem(x) for x in y] for y in punctuation_lowercase]
punctuation_lowercase_stemmed_dash = [[x.replace(' ', '-') for x in y] for y in punctuation_lowercase_stemmed]
json_ingredients_cleaned = [' '.join(x) for x in punctuation_lowercase_stemmed_dash]

# grab cuisine variables
json_cuisine = [x['cuisine'] for x in json_data]

# turn targets into ints
lb = LabelEncoder()
y = lb.fit_transform(json_cuisine)

# implement tfid
tfidf = TfidfVectorizer(use_idf=True, ngram_range=(1,3))
X = tfidf.fit_transform(json_ingredients_cleaned)

model = SVC(
	 		   kernel='rbf',
                C = 100,
	 		   gamma=.01,
	 		   tol=0.001,
	      	   verbose=False,
	      	   max_iter=-1,
            )


## Model Tuning
#parameters = {"gamma":[0.01, 0.5, 0.1, 2, 5], "C":[1, 10, 100, 1000]}
#grid_search = GridSearchCV(model, param_grid=parameters, cv=5, n_jobs=-1)
#grid_search.fit(X, y)
#print(grid_search.best_score_)
#print(grid_search.best_params_)
#---------------------------------
## results of grid_search
#best score: .776889
#best params: C=100, gamma=0.01

model.fit(X, y)

# now, we observe the test data
# THIS PROCESS COULD HAVE BEEN OPTIMIZED BY CREATING A FUNCTION
# SHOULD BE DONE TO FOLLOW DRY PRACTICES
with open('../input/whats-cooking-kernels-only/test.json') as json_file:
    json_data_test = json.load(json_file)

len(json_data_test)

json_ingredients_test = [x['ingredients'] for x in json_data_test]

# make lowercase, clear punctuation, and stem
# dashes are added so ngram isn't confused
lowercase_test = [[x.lower() for x in y] for y in json_ingredients_test]
punctuation_lowercase_test = [[re.sub(r'([^\s\w]|_)+', '', x) for x in y] for y in lowercase_test]
punctuation_lowercase_stemmed_test = [[sno.stem(x) for x in y] for y in punctuation_lowercase_test]
punctuation_lowercase_stemmed_dash_test = [[x.replace(' ', '-') for x in y] for y in punctuation_lowercase_stemmed_test]
json_ingredients_cleaned_test = [' '.join(x) for x in punctuation_lowercase_stemmed_dash_test]

X_test = tfidf.transform(json_ingredients_cleaned_test)


# Predictions
y_test = model.predict(X_test)
y_pred = lb.inverse_transform(y_test)

test_id = [doc['id'] for doc in json_data_test]
df = pd.DataFrame({'id':test_id, 'cuisine':y_pred}, columns=['id','cuisine'])
df.to_csv('cuisine_output.csv', index=False)

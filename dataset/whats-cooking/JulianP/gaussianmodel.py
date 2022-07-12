# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:15:01 2015

@author: JNPD
"""

# Whats cooking?

import numpy as np
import json
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


train = pd.read_json('../input/train.json')
train.head()

#Initalize a CountVectorizer only considering the top 2000 features. 
#Then Extract the ingredients and convert them to a single list of recipes called words_list
vectorizer = CountVectorizer(max_features = 4000)
ingredients = train['ingredients']
words_list = [' '.join(x) for x in ingredients]

#create a bag of words and convert to a array and then print the shape
bag_of_words = vectorizer.fit(words_list)
bag_of_words = vectorizer.transform(words_list).toarray()

gnb = GaussianNB()	# create the classifier object
gnbfit = gnb.fit(bag_of_words, train["cuisine"])	# fit (/train) the classifier to the data

#Now read the test json file in 
test = pd.read_json('../input/test.json')
test.head()

#Do the same thing we did with the training set and create a array using the count vectorizer. 
test_ingredients = test['ingredients']
test_ingredients_words = [' '.join(x) for x in test_ingredients]
test_ingredients_array = vectorizer.transform(test_ingredients_words).toarray()
result = gnb.predict(test_ingredients_array)
output = pd.DataFrame( data={"id":test["id"], "cuisine":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )
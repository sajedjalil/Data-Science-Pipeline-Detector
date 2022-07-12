#!/usr/bin/env python
#Import Packages we will be using 
import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = pd.read_json('../input/train.json')
train.head()

#Initalize a CountVectorizer only considering the top 2000 features. 
#Then Extract the ingredients and convert them to a single list of recipes called words_list
vectorizer = CountVectorizer(max_features = 2000)
ingredients = train['ingredients']
words_list = [' '.join(x) for x in ingredients]

#create a bag of words and convert to a array and then print the shape
bag_of_words = vectorizer.fit(words_list)
bag_of_words = vectorizer.transform(words_list).toarray()
print(bag_of_words.shape)

#Initilize a random forest classifier with 500 trees and fit it with the bag of words we created 
forest = RandomForestClassifier(n_estimators = 500) 
forest = forest.fit( bag_of_words, train["cuisine"] )

#Now read the test json file in 
test = pd.read_json('../input/test.json')
test.head()

#Do the same thing we did with the training set and create a array using the count vectorizer. 
test_ingredients = test['ingredients']
test_ingredients_words = [' '.join(x) for x in test_ingredients]
test_ingredients_array = vectorizer.transform(test_ingredients_words).toarray()

# Use the random forest to make cusine predictions
result = forest.predict(test_ingredients_array)

result

# Copy the results to a pandas dataframe with an "id" column and
# a "cusine" column
output = pd.DataFrame( data={"id":test["id"], "cuisine":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )
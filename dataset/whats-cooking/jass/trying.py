#Load the packages that will be needed
import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
 
#Load the train json file using pandas read_json command
traindf = pd.read_json("../input/train.json")
traindf.head()
print (traindf)
vectorizer = CountVectorizer(max_features = 1000)
#Convert the list of ingredients to one big list 
words_list = [' , '.join(z).strip() for z in traindf['ingredients']]
 
#create a bag of words and convert to a array and then print the shape
bag_of_words = vectorizer.fit(words_list)
bag_of_words = vectorizer.transform(words_list).toarray()
print(bag_of_words.shape)
#Initilize a random forest classifier with 200trees and fit it with the bag of words we created 
forest = RandomForestClassifier(n_estimators = 200) 
forest = forest.fit( bag_of_words, traindf["cuisine"] )
#Now read the test json file in 
test = pd.read_json("../input/test.json")
test.head()
print (test)
#Do the same thing we did with the training set and create a array using the count vectorizer. 
test_ingredients_words = [' , '.join(z).strip() for z in test['ingredients']] 
test_ingredients_array = vectorizer.transform(test_ingredients_words).toarray()
# Use the random forest to make cusine predictions
result = forest.predict(test_ingredients_array)
# Copy the results to a pandas dataframe with an "id" column and
# a "cusine" column
print (result)
output = pd.DataFrame( data={"id":test["id"], "cuisine":result} )# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3)

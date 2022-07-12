
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import os
import re
from sklearn.ensemble import AdaBoostClassifier

traindf = pd.read_json("../input/train.json")
traindf.head()
print (traindf)
vectorizer = CountVectorizer(max_features = 1000)

words_list = [' , '.join(z).strip() for z in traindf['ingredients']]

bag_of_words = vectorizer.fit(words_list)
bag_of_words = vectorizer.transform(words_list).toarray()
print(bag_of_words.shape)
 
forest = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
forest = forest.fit( bag_of_words, traindf["cuisine"] )
 
test = pd.read_json("../input/test.json")
test.head()
#print (test)
 
test_ingredients = [' , '.join(z).strip() for z in test['ingredients']] 
array_test = vectorizer.transform(test_ingredients).toarray()
result = forest.predict(array_test)
print (result)
output = pd.DataFrame( data={"id":test["id"], "cuisine":result} )# Use pandas to write the comma-separated output file
print (output)
#output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3)
result = [(ref == res, ref, res) for (ref, res) in zip(traindf["cuisine"], forest.predict(array_test))]
accuracy_learn = sum (r[0] for r in result) / float ( len(result) )
print(sum)
#print (clf2)
print('Accuracy : ', accuracy_learn)
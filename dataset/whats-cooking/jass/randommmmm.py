
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import os
import re
from sklearn.metrics import accuracy_score
traindf = pd.read_json("../input/train.json")
traindf.head()
print (traindf)
vectorizer = CountVectorizer(max_features = 1000)

words_list = [' , '.join(z).strip() for z in traindf['ingredients']]

bag_of_words = vectorizer.fit(words_list)
bag_of_words = vectorizer.transform(words_list).toarray()
print(bag_of_words.shape)
 
forest = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None) 
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
#print(accuracy_score(test_ingredients,result,normalize=False))
#output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3)
result = [(ref == res, ref, res) for (ref, res) in zip(traindf["cuisine"], forest.predict(array_test))]
accuracy_learn = sum (r[0] for r in result) / float ( len(result) )
print(sum)
#print (clf2)
print('Accuracy : ', accuracy_learn)

# Standard Imports
import pandas as pd
import numpy as np

# Performance
from time import time

# Machine Learning
from sklearn.ensemble import RandomForestClassifier

# Helper
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# Load in the Data
train = pd.read_json('../input/train.json')

# Extract the Unique Ingredients
words = [' '.join(item) for item in train.ingredients]

# Construct the Bag of Words
vec = CountVectorizer(max_features=2000)
bag_of_words = vec.fit(words).transform(words).toarray()

## Random Forest Classification
random_forest = RandomForestClassifier(n_estimators=200)

# Recored the time it takes to perform the search
start = time()
random_forest.fit(bag_of_words, train.cuisine)
print("RandomForest Training finished in %.2f" % (time() - start))

# Basic Evaluation on Training Set
start = time()
train_pred = cross_val_predict(random_forest, bag_of_words, train.cuisine, cv=2)
print("RandomForest Evaluation finished in %.2f" % (time() - start))

# Display Accuracy
print("Accuracy: ", accuracy_score(train.cuisine, train_pred))

# Load in Testing Data
test = pd.read_json('../input/test.json')

# Create test Bag of Words
test_words = [' '.join(item) for item in test.ingredients]
test_bag = vec.transform(test_words).toarray()

# Run the Prediction
result = random_forest.predict(test_bag)

output = pd.DataFrame(data={"id":test.id, "cuisine":result})

output.to_csv("submission.csv", index=False, quoting=3)
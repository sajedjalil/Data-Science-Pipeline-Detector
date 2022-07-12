# Standard Imports
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.ensemble import AdaBoostClassifier

# Helper
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# Load in the Data
train = pd.read_json('../input/train.json')

# Extract the Unique Ingredients
words = [' '.join(item) for item in train.ingredients]

# Construct the Bag of Words
vec = CountVectorizer(max_features=2000)
bag_of_words = vec.fit(words).transform(words).toarray()

# AdaBoost Classification
pipeline = make_pipeline(AdaBoostClassifier(n_estimators=200))
pipeline.fit(bag_of_words, train.cuisine)

# Basic Evaluation on Training Set
train_pred = pipeline.predict(bag_of_words)

# Display Accuracy
print("Accuracy: ", accuracy_score(train.cuisine, train_pred))

# Load in Testing Data
test = pd.read_json('../input/test.json')

# Create test Bag of Words
test_words = [' '.join(item) for item in test.ingredients]
test_bag = vec.transform(test_words).toarray()

# Run the Prediction
result = pipeline.predict(test_bag)

output = pd.DataFrame(data={"id":test.id, "cuisine":result})

output.to_csv("submission.csv", index=False, quoting=3)

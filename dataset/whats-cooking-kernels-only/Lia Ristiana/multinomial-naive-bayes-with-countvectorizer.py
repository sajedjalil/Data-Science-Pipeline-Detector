# This is my first kernel on Kaggle, so I am still figuring out how things work around here.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

import os

train_json = pd.read_json("../input/train.json")
test_json = pd.read_json("../input/test.json")

train_json.to_csv("train.csv", index=False)
test_json.to_csv("test.csv", index=False)

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# READ TRAINING DATA AND SEPARATE INTO X AND y
X_train = train_df['ingredients']
y_train = train_df['cuisine']

X_test = test_df['ingredients']

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# MNB CLASSIFICATION
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
mnb_prediction = mnb.predict(X_test)

mnb_results = np.array(list(zip(test_df['id'],mnb_prediction)))
mnb_results = pd.DataFrame(mnb_results, columns=['id', 'cuisine'])
mnb_results.to_csv('mnb_vectorized.csv', index = False)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer

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

vectorizer = CountVectorizer(min_df=4)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# LOGISTIC REGRESSION
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
lr_prediction = logreg.predict(X_test)

lr_results = np.array(list(zip(test_df['id'],lr_prediction)))
lr_results = pd.DataFrame(lr_results, columns=['id', 'cuisine'])
lr_results.to_csv('lr_vectorized.csv', index = False)
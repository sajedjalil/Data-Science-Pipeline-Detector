# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings #to ignore any warnings comes
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#loading training and testing datasets
print("loading training and testing datasets")
train_data=pd.read_csv('../input/labeledTrainData.tsv',header = 0, delimiter = '\t')
test_data=pd.read_csv('../input/testData.tsv',header = 0, delimiter = '\t')

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

all_text=train_data['review']
train_text=train_data['review']
y=train_data['sentiment']

print("Create a bag of words from the training set cleaning")
# Create a bag of words from the training set
word_vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', analyzer='word',
    token_pattern=r'\w{1,}', stop_words='english', ngram_range=(1, 1), max_features=10000)
word_vectorizer.fit(all_text)

train_word_features = word_vectorizer.transform(train_text)

char_vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', analyzer='char',
    stop_words='english', ngram_range=(2, 6), max_features=50000)
char_vectorizer.fit(all_text)

train_char_features = char_vectorizer.transform(train_text)

train_features = hstack([train_char_features, train_word_features])

# Train a LogisticRegression algorithm using the bag of words
print("Train a LogisticRegression algorithm using the bag of words")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_features, y,test_size=0.3,random_state=101)

# This may take a few minutes to run
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

# Fit the LogisticRegression to the training set, using the bag of words as
# features and the sentiment labels as the response variable
print("Fit the LogisticRegression to the training set, using the bag of words")
lr.fit(X_train,y_train)
preds=lr.predict(X_test)

# measuring parameters of the model like precison, f1score, confusuion matix
print("measuring metrics")
from sklearn.metrics import classification_report, confusion_matrix
print("classification_report")
print(classification_report(preds,y_test))
print("confusion_matrix")
print(confusion_matrix(preds,y_test))

train_text = train_data['review']
test_text = test_data['review']
all_text = pd.concat([train_text, test_text])

print("Create a bag of words from the testing set and cleaning")
word_vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', analyzer='word',
    token_pattern=r'\w{1,}', stop_words='english', ngram_range=(1, 1), max_features=10000)
word_vectorizer.fit(all_text)

train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', analyzer='char',
    stop_words='english', ngram_range=(2, 6), max_features=50000)
char_vectorizer.fit(all_text)

train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

print("Fit the LogisticRegression to the tesing set, using the bag of words")
lr=LogisticRegression(C=0.1,solver='sag')
lr.fit(train_features,y)
final_preds=lr.predict(test_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
test_data['sentiment']=final_preds
test_data=test_data[['id','sentiment']]

# Use pandas to write the comma-separated output file
print("Using pandas writing the comma-separated output file as 'Result.csv'")
test_data.to_csv('Result.csv',index=False)
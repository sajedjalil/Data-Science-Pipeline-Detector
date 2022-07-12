import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import textblob

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import re
import pickle
from scipy.sparse import hstack

from sklearn import preprocessing, model_selection, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, make_scorer,roc_curve, roc_auc_score

import os
print(os.listdir("../input/"))

# Load Data
print("Loading data...")

train = pd.read_csv('../input/labeledTrainData.tsv', sep="\t")
print("Train shape:", train.shape)
test = pd.read_csv('../input/testData.tsv', sep="\t")
print("Test shape:", test.shape)

sample = pd.read_csv('../input/sampleSubmission.csv', sep=",")


print(train.head())
print(test.head())
print(sample.head())

print("Value counts of sentiment class", train['sentiment'].value_counts()) # balanced dataset

# Check the first review
print('The first review is:\n\n',train["review"][0])

# clean description
print("Cleaning train data...\n")
train['review'] = train['review'].map(lambda x: BeautifulSoup(x).get_text())
print("Cleaning test data...")
test['review'] = test['review'].map(lambda x: BeautifulSoup(x).get_text())


# function to clean data

stops = set(stopwords.words("english"))
def cleanData(text, lowercase = False, remove_stops = False, stemming = False):
    txt = str(text)
    txt = re.sub(r'[^A-Za-z0-9\s]',r'',txt)
    txt = re.sub(r'\n',r' ',txt)
    
    if lowercase:
        txt = " ".join([w.lower() for w in txt.split()])
        
    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in stops])
    
    if stemming:
        st = PorterStemmer()
        txt = " ".join([st.stem(w) for w in txt.split()])

    return txt
    
y = train['sentiment']

# Bag of Words (word based)
ctv_word = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',min_df = 200, max_features=5000,
            ngram_range=(1,2), stop_words = 'english')

print("Fitting Bag of Words Model on words...\n")
# Fitting CountVectorizer to both training and test sets
ctv_word.fit(list(train['review']) + list(test['review']))
train_ctv_word =  ctv_word.transform(train['review']) 
test_ctv_word = ctv_word.transform(test['review'])

print("Fitting Bag of Words Model on characters...\n")

# Bag of words (charater based)
ctv_char = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode',analyzer='char',
    stop_words='english', ngram_range=(2, 6), max_features=10000)

# Fitting CountVectorizer to both training and test sets
ctv_char.fit(list(train['review']) + list(test['review']))
train_ctv_char =  ctv_char.transform(train['review']) 
test_ctv_char = ctv_char.transform(test['review'])

# TF - IDF (words)

print("Fitting TF-IDF Model on words...\n")
tfv_word = TfidfVectorizer(min_df=150,  max_features= 5000, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1,2),
            stop_words = 'english')

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv_word.fit(list(train['review']) + list(test['review']))
train_tfv_word =  tfv_word.transform(train['review'])
test_tfv_word = tfv_word.transform(test['review'])

# TF-IDF(char)
print("Fitting TF - IDF Model on characters...\n")
tfv_char = TfidfVectorizer(sublinear_tf=True,strip_accents='unicode',analyzer='char',
    stop_words='english',ngram_range=(2, 6),max_features=10000)
tfv_char.fit(list(train['review']) + list(test['review']))
train_tfv_char = tfv_char.transform(train['review'])
test_tfv_char = tfv_char.transform(test['review'])

print("Combining Bag of words for words and characters...\n")
# bag of words for training set (words + char)
train_bow = hstack([train_ctv_word, train_ctv_char])
test_bow = hstack([test_ctv_word, test_ctv_char])

print("Combining TF-IDF for words and characters...\n")

# TF-IDF for test set (words + char)
train_tfidf = hstack([train_tfv_word, train_tfv_char])
test_tfidf = hstack([test_tfv_word, test_tfv_char])

clf_lr = LogisticRegression() # Logistic Regression Model

## 5-fold cross validation
print(cross_val_score(clf_lr, train_tfidf, y, cv=5, scoring=make_scorer(f1_score)))

"""
We can see that we are achieving an validation accuracy of 89%. Which is really interesting
"""
# Fit the logistic regression model
clf_lr.fit(train_tfidf,y)

# Make predictions on test data
preds = clf_lr.predict(test_tfidf)

# Make submission

sample['sentiment'] = preds
sample = sample[['id','sentiment']]
sample.to_csv('submissions.csv',index=False)
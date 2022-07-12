# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import gc
pd.set_option('display.max_colwidth', 100)

gc.collect()


# Loading the dataset
train = pd.read_csv('../input/train.tsv', sep='\t')
print(train.shape)
train.head()

test = pd.read_csv('../input/test.tsv', sep='\t')
print(test.shape)
test.head()

sub = pd.read_csv('../input/sampleSubmission.csv')
sub.head()

# Adding sentiment column to test dataset and join test and train for preprocessing
test['Sentiment'] = -999
test.head()

df = pd.concat([train,test], ignore_index=True)
print(df.shape)
df.tail()

del train,test
gc.collect()

# Cleaning reviews
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
stemmer = SnowballStemmer('english')
lemma = WordNetLemmatizer()
from string import punctuation
import re
from bs4 import BeautifulSoup

def clean_review(review_col):
    review_corpus = []
    for i in range(0, len(review_col)):
        review = str(review_col[i])
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = [lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review = ''.join(review)
        review_corpus.append(review)
    return review_corpus

df['clean_review'] = clean_review(df.Phrase.values)
df.head()

# Seperating test and train set
df_train=df[df.Sentiment!=-999]
df_train.shape

df_test=df[df.Sentiment==-999]
df_test.drop('Sentiment',axis=1,inplace=True)
print(df_test.shape)
df_test.head()

del df
gc.collect()

# Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(ngram_range=(1,2),max_df=0.95,min_df=10,sublinear_tf=True)

c2_train=tfidf.fit_transform(df_train.clean_review).toarray()
print(c2_train.shape)
c2_test=tfidf.transform(df_test.clean_review).toarray()
print(c2_test.shape)

# One Hot Encoding dependent variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
y=le.fit_transform(df_train.Sentiment.values)
y.shape
del df_train,df_test
gc.collect()

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression() 
lr.fit(c2_train,y)

y_pred=lr.predict(c2_test)

sub.Sentiment=y_pred
sub.head()
sub.to_csv('submission.csv',index=False)

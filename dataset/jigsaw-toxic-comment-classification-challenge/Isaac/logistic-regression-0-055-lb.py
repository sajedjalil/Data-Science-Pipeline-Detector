# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import os
from sklearn import preprocessing
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score , roc_auc_score , log_loss
import sklearn.linear_model as lm
from sklearn.model_selection import GridSearchCV
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, log_loss
from numpy import linalg as LA
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
import datetime as dt
import nltk.stem  

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# StemmedTfidfVectorizer
english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

def clean_text( text ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    #text = BeautifulSoup(review,'html.parser').get_text()
    #
    # 2. Remove non-letters
    text = re.sub("[^A-za-z0-9^,?!.\/'+-=]"," ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " _exclamationmark_ ", text)
    text = re.sub(r"\?", " _questionmark_ ", text)
    #
    return text


def build_data_set(ngram=3,stem=False,max_features=2000,min_df=2,remove_stopwords=True):
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    test.fillna('__NA__',inplace=True)
    ## 
    clean_train_comments = []
    for i in range(train.shape[0]):
        clean_train_comments.append( clean_text(train["comment_text"][i]) )
    #print(">>> processing test set ...")
    for i in range(test.shape[0]):
        clean_train_comments.append( clean_text(test["comment_text"][i]) )
    qs = pd.Series(clean_train_comments).astype(str)
    if not stem:
        # 1-gram / no-stem
        vect = TfidfVectorizer(analyzer=u'word',stop_words='english',min_df=min_df,ngram_range=(1, ngram),max_features=max_features)
        ifidf_vect = vect.fit_transform(qs) 
        #print("ifidf_vect:", ifidf_vect.shape)
        X = ifidf_vect.toarray()
        X_train = X[:train.shape[0]]
        X_test = X[train.shape[0]:]
    else:
        vect_stem = StemmedTfidfVectorizer(analyzer=u'word',stop_words='english',min_df=min_df,ngram_range=(1, ngram),max_features=max_features)
        ifidf_vect_stem = vect_stem.fit_transform(qs)
        #print("ifidf_vect_stem:", ifidf_vect_stem.shape)
        X = ifidf_vect_stem.toarray()
        X_train = X[:train.shape[0]]
        X_test = X[train.shape[0]:]
    Y_train = train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
    assert Y_train.shape[0] == X_train.shape[0]
    del train, test
    return X_train,X_test,Y_train
    


#--------------------------- Main()
labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
params = {
    'toxic': {'ngrams': 1, 'stem': True, 'max_features': 10000, 'C': 10 } , 
    'threat': {'ngrams': 1, 'stem': False, 'max_features': 10000, 'C': 10 } , 
    'severe_toxic': {'ngrams': 1, 'stem': True, 'max_features': 5000, 'C': 1.2 } , 
    'obscene': {'ngrams': 1, 'stem': True, 'max_features': 10000, 'C': 10 } , 
    'insult': {'ngrams': 1, 'stem': True, 'max_features': 10000, 'C': 1.2 } , 
    'identity_hate': {'ngrams': 1, 'stem': True, 'max_features': 10000, 'C': 10 } 
}

sample_submission = pd.read_csv("../input/sample_submission.csv")

# proc
t0 = dt.datetime.now()

for label in labels:
    print(">>> processing ",label)
    X_train,X_test,Y_train = build_data_set(ngram=params[label]['ngrams'],
                                            stem=params[label]['stem'],
                                            max_features=params[label]['max_features'],
                                            min_df=2,remove_stopwords=True)
    Y_train_lab = Y_train[label]
    clf = lm.LogisticRegression(C=params[label]['C'])
    clf.fit(X_train, Y_train_lab)
    del X_train, Y_train_lab
    pred_proba = clf.predict_proba(X_test)[:, 1]
    del X_test
    sample_submission[label] = pred_proba
    sample_submission.to_csv("baseline_linear.csv", index=False)


t1 = dt.datetime.now()
print('Total time: %i seconds' % (t1 - t0).seconds)
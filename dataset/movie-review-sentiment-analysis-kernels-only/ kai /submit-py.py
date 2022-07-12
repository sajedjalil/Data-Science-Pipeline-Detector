# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 21:55:20 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 03:50:35 2018
@author: kaixiong
"""
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.linear_model import LogisticRegression  
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def get_wordnet_pos(treebank):
    if treebank.startswith('J'):
        return wordnet.ADJ
    elif treebank.startswith('V'):
        return wordnet.VERB
    elif treebank.startswith('N'):
        return wordnet.NOUN
    elif treebank.startswith('R'):
        return wordnet.ADV
    else:
        return None
#tag for lemmatization
        
def lemmatize_sentence(sentence):
    from nltk.stem import WordNetLemmatizer
    res = []
    lemmatizer = WordNetLemmatizer()
    for word, pos in pos_tag(word_tokenize(sentence)):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return res
#lemmatization
    
def remove_stopwords(train):
 stop_words = set(stopwords.words('english'))
 remove_list=[]
 for deleted_words in train:
     if deleted_words not in stop_words:
             pass
     else:
            remove_list.append(deleted_words)           
 for deleted_words in remove_list:
      if deleted_words in remove_list:
          train.remove(deleted_words)
      else:
          pass
#remove stopwords
          
train = pd.read_csv('../input/train.tsv',sep ='\t')    
test = pd.read_csv('../input/test.tsv', sep="\t")
sample = pd.read_csv('../input/sampleSubmission.csv')

train['Phrase'] = train['Phrase'].apply(lambda x: x.lower())
train['Phrase'] = train['Phrase'].apply(lambda x: lemmatize_sentence(x))
train['Phrase'].apply(remove_stopwords)
train['Phrase'] = train['Phrase'].apply(lambda x:  ' '.join(x))

test['Phrase'] = test['Phrase'].apply(lambda x: x.lower())
test['Phrase'] = test['Phrase'].apply(lambda x: lemmatize_sentence(x))
test['Phrase'].apply(remove_stopwords)
test['Phrase'] = test['Phrase'].apply(lambda x:  ' '.join(x))

transformer =TfidfVectorizer(ngram_range=(1,3))
train_tfidf = transformer.fit_transform(train['Phrase'])
test_tfidf = transformer.transform(test['Phrase'])

print(train_tfidf.shape)
print(test_tfidf.shape)

clf = LogisticRegression(C = 2.6,solver='sag')
clf.fit(train_tfidf,train['Sentiment'])
results = clf.predict(test_tfidf)
accuracy = accuracy_score(sample['Sentiment'], results)

sample['Sentiment'] = results
sample.to_csv('submission',index = False)

#logic regression -> predict test data


       
       


       



    

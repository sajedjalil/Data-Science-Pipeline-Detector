# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 00:23:13 2018

@author: arnab
"""

import pandas as pd
from pandas import DataFrame
# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Word Stemmer

from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
stemmer = SnowballStemmer("english", ignore_stopwords=True)




train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

train.head()

train_text = train['comment_text']


test_text = test['comment_text']



#Clean some unwanted Characters - Too Slow / Need to Replace with Regex
#train_text = [train_text.replace('\n\n',' ') for word in train_text]
#test_text1 = [test_text1.replace('\n\n',' ') for word in test_text1]
#reg = re.compile(r'\n\n')
#rep=''
#train_text = [re.sub(reg, rep, train_text) for word in train_text]



test_text1=test_text.copy()
train_text1 = train_text.copy()

# Stem the results to the root word

for i in range(1,train_text.shape[0]):
    stem_text=stemmer.stem(train_text[i])
    train_text1[i]=stem_text

for i in range(1,test_text.shape[0]):
    stem_text=stemmer.stem(test_text[i])
    test_text1[i]=stem_text


#train_texts = [[stemmer.stem(word) for word in text.split()] for text in train_text]

#test_texts = [[stemmer.stem(word) for word in text.split()] for text in test_text]
class_names = ['toxic', 'threat','severe_toxic', 'obscene',  'insult', 'identity_hate']

pl=Pipeline([('vect', TfidfVectorizer(strip_accents='unicode',stop_words = 'english',analyzer='word',ngram_range=(1,2))),
              ('chi',SelectKBest(chi2,k=50000)),
              ('clf',SVC(max_iter=3000,probability=True))]) # SVC Performance is slow

# LR is preferred for computational speed
pl_LR=Pipeline([('vect', TfidfVectorizer(strip_accents='unicode',stop_words = 'english',analyzer='word',ngram_range=(1,2))),
              ('chi',SelectKBest(chi2,k=15000)),
              ('clf',LogisticRegression(max_iter=5000))])
    
    

# The following loop is to check the Model Scores

test2=pd.concat([pd.DataFrame(test['id']) , pd.DataFrame(test_text1)], axis=1)

#test3=test2[:3] # For Testing of the Loop

for column in class_names:
    print(column)
    model=pl_LR.fit(train_text1,train[column])
    
    #for row in test2.iterrows():
    test2[column]=model.predict_proba(test2['comment_text'])[:,1]
             #test3[column]=model.predict(test3['comment_text'])
   
# SUBMISSION FILE READY
#submission = test2[['id','threat']]           
submission = pd.DataFrame(test2[['id','toxic','threat','severe_toxic', 'obscene',  'insult', 'identity_hate']])
submission.to_csv('submission.csv', index=False)

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression

# load data file into Python
train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json") 

# clean data
train['ingredients_clean_string'] = [' , '.join(z).strip() for z in train['ingredients']]  
test['ingredients_clean_string'] = [' , '.join(z).strip() for z in test['ingredients']]

# further clean data and extract information through word lemmatization
train['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) 
                                         for line in lists]).strip() for lists in train['ingredients']]       
test['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) 
                                          for line in lists]).strip() for lists in test['ingredients']]       

# create corpus based on newly processed data
train_corpus = train['ingredients_string']
test_corpus = test['ingredients_string']

# convert a collection of raw documents to a matrix of TF-IDF features
train_vectorizer = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
test_vectorizer = TfidfVectorizer(stop_words='english')

# transform the corpus to a dense matrix representation
train_tfidf=train_vectorizer.fit_transform(train_corpus).todense()
test_tfidf=train_vectorizer.transform(test_corpus)


# prepare data for prediction
train_predictor = train_tfidf
test_predictor = test_tfidf

train_target = train['cuisine']


# build Linear Support Vector Classification model
# set penalty parameter as 0.8 with standard penaliation l2
# select the algorithm to solve primal optiomization problem
classifier = LinearSVC(C=0.80, penalty="l2", dual=False)

# model = LinearSVC()
model = LogisticRegression()

# process exhaustive search over specified parameter values for the model
parameters = {'C':[1, 10]}
classifier = grid_search.GridSearchCV(model, parameters)

# fit classification model to data
classifier=classifier.fit(train_predictor,train_target)

# make prediction
prediction=classifier.predict(test_predictor)

# assign predicted values to cuisine in TEST set
test['cuisine'] = prediction

# write csv file (no index for submission)
test[['id','cuisine' ]].to_csv("LogisticRegression.csv",index=False)
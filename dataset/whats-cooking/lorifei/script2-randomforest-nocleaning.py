import pandas as pd
from pandas import Series, DataFrame
import json
import numpy as np
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer

traindf = pd.read_json("../input/train.json")
testdf = pd.read_json("../input/test.json")

stemmer = WordNetLemmatizer()
def stem(Ing):
    i= ""
    Ing = Ing.split(" ")
    for n in Ing:
        i = i + stemmer.lemmatize(n,pos='n')
    return i

def Tostring(Ing):
    i = ""
    for n in Ing:
        i = i + n + " , "
    return i

traindf['ingredients_str'] = traindf['ingredients'].apply(lambda x: Tostring(x)) 
traindf['ingredients'] = traindf['ingredients_str'].apply(lambda x: stem(x))

testdf['ingredients_str'] = testdf['ingredients'].apply(lambda x: Tostring(x)) 
testdf['ingredients'] = testdf['ingredients_str'].apply(lambda x: stem(x))

Corpus1 = traindf['ingredients']
vectorizer = TfidfVectorizer(ngram_range = ( 1 , 1 ),analyzer="word", max_df = 0.6, min_df = 0.01,token_pattern=r'\w+')
tfidf1=vectorizer.fit_transform(Corpus1).todense()

Corpus2 = testdf['ingredients']
tfidf2=vectorizer.transform(Corpus2)

y = traindf['cuisine']
clf = ensemble.RandomForestClassifier(n_estimators = 100)
clf.fit(tfidf1, y)

clf.predict(tfidf2)

testdf['cuisine'] = clf.predict(tfidf2)

testdf[['id' , 'cuisine' ]].to_csv("submission_randomforest.csv",index=False)
Script2 = pd.read_csv("submission_randomforest.csv")
Script2


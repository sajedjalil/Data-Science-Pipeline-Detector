import pandas as pd
from pandas import Series, DataFrame
import json
import numpy as np
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
import re
from nltk.stem import WordNetLemmatizer
from collections import Counter

traindf = pd.read_json('../input/train.json')
testdf = pd.read_json('../input/test.json')

strip = lambda x: ','.join(x).strip() 
traindf['ingredients'] = traindf['ingredients'].map(strip)
testdf['ingredients'] = testdf['ingredients'].map(strip)

lower = lambda x: str.lower(x)
traindf['ingredients'] = traindf['ingredients'].map(lower)
testdf['ingredients'] = testdf['ingredients'].map(lower)

salt = lambda x: re.sub(r'\bsalt\b',' ', x)
traindf['ingredients'] = traindf['ingredients'].map(salt)
testdf['ingredients'] = testdf['ingredients'].map(salt)

dash = lambda x: re.sub(r"(?i)(?<=[A-Z])-(?=[A-Z])"," ", x)
traindf['ingredients'] = traindf['ingredients'].map(dash)
testdf['ingredients'] = testdf['ingredients'].map(dash)

char_only = lambda x: re.sub('[^a-zA-Z0-9 \n\,]',' ', x)
traindf['ingredients'] = traindf['ingredients'].map(char_only)
testdf['ingredients'] = testdf['ingredients'].map(char_only)

water = lambda x: re.sub(r'\bwater\b',' ', x)
traindf['ingredients'] = traindf['ingredients'].map(water)
testdf['ingredients'] = testdf['ingredients'].map(water)

eggs = lambda x: re.sub(r'\blarge eggs\b', 'eggs', x)
traindf['ingredients'] = traindf['ingredients'].map(eggs)
testdf['ingredients'] = testdf['ingredients'].map(eggs)

pepper = lambda x: re.sub(r'\bground black pepper\b', 'black pepper', x)
traindf['ingredients'] = traindf['ingredients'].map(pepper)
testdf['ingredients'] = testdf['ingredients'].map(pepper)

olive_oil = lambda x: re.sub(r'\bextravirgin olive oil\b', 'olive oil', x)
traindf['ingredients'] = traindf['ingredients'].map(olive_oil)
testdf['ingredients'] = testdf['ingredients'].map(olive_oil)

butter = lambda x: re.sub(r'\bunsalted butter\b', 'butter', x)
traindf['ingredients'] = traindf['ingredients'].map(butter)
testdf['ingredients'] = testdf['ingredients'].map(butter)

flour = lambda x: re.sub(r'\ballpurpose flour\b', 'flour', x)
traindf['ingredients'] = traindf['ingredients'].map(flour)
testdf['ingredients'] = testdf['ingredients'].map(flour)

ginger = lambda x: re.sub(r'\bfresh ginger\b', 'ginger', x)
traindf['ingredients'] = traindf['ingredients'].map(ginger)
testdf['ingredients'] = testdf['ingredients'].map(ginger)

lime_juice = lambda x: re.sub(r'\bfresh lime juice\b', 'lemon juice', x)
traindf['ingredients'] = traindf['ingredients'].map(lime_juice)
testdf['ingredients'] = testdf['ingredients'].map(lime_juice)

lemon_juice = lambda x: re.sub(r'\bfresh lemon juice\b', 'lemon juice', x)
traindf['ingredients'] = traindf['ingredients'].map(lemon_juice)
testdf['ingredients'] = testdf['ingredients'].map(lemon_juice)

lemon = lambda x: re.sub(r'\blime\b', 'lemon', x)
traindf['ingredients'] = traindf['ingredients'].map(lemon)
testdf['ingredients'] = testdf['ingredients'].map(lemon)

replace_whitespace = lambda x: re.sub("(?m)^\s+", "", x)
traindf['ingredients'] = traindf['ingredients'].map(replace_whitespace)
testdf['ingredients'] = testdf['ingredients'].map(replace_whitespace)

def Tostring(Ing):
    i = ""
    for n in Ing:
        i = i + n + " , "
    return i
    
traindf['ingredients_str'] = traindf['ingredients'].apply(lambda x: Tostring(x)) 

testdf['ingredients_str'] = testdf['ingredients'].apply(lambda x: Tostring(x)) 

Corpus1 = traindf['ingredients']
vectorizer1 = TfidfVectorizer(stop_words='english', ngram_range = ( 1 , 1 ),analyzer="word", max_df = 0.54,  binary=False , token_pattern=r'\w+')
tfidf1=vectorizer1.fit_transform(Corpus1).todense()

Corpus2 = testdf['ingredients']
tfidf2=vectorizer1.transform(Corpus2)

y = traindf['cuisine']
clf = ensemble.RandomForestClassifier(n_estimators = 600)
clf.fit(tfidf1, y)

clf.predict(tfidf2)
testdf['cuisine'] = clf.predict(tfidf2)

testdf[['id' , 'cuisine' ]].to_csv("submission_rf.csv",index=False)
A = pd.read_csv("submission_rf.csv")
A
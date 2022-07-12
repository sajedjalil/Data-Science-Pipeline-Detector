import pandas as pd
from pandas import Series, DataFrame
import json
import numpy as np
import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.linear_model import LogisticRegression as LR
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV

traindf = pd.read_json('../input/train.json')
testdf = pd.read_json('../input/test.json')

strip = lambda x: ','.join(x).strip() 
traindf['ingredients'] = traindf['ingredients'].map(strip)
testdf['ingredients'] = testdf['ingredients'].map(strip)

lower = lambda x: str.lower(x)
traindf['ingredients'] = traindf['ingredients'].map(lower)
testdf['ingredients'] = testdf['ingredients'].map(lower)

char_only = lambda x: re.sub('[^a-zA-Z0-9 \n\,]',' ', x)
traindf['ingredients'] = traindf['ingredients'].map(char_only)
testdf['ingredients'] = testdf['ingredients'].map(char_only)

eggs = lambda x: re.sub(r'\blarge eggs\b', 'eggs', x)
traindf['ingredients'] = traindf['ingredients'].map(eggs)
testdf['ingredients'] = testdf['ingredients'].map(eggs)

pepper = lambda x: re.sub(r'\bground black pepper\b', 'black pepper', x)
traindf['ingredients'] = traindf['ingredients'].map(pepper)
testdf['ingredients'] = testdf['ingredients'].map(pepper)

olive_oil = lambda x: re.sub(r'\bextra virgin olive oil\b', 'olive oil', x)
traindf['ingredients'] = traindf['ingredients'].map(olive_oil)
testdf['ingredients'] = testdf['ingredients'].map(olive_oil)

butter = lambda x: re.sub(r'\bunsalted butter\b', 'butter', x)
traindf['ingredients'] = traindf['ingredients'].map(butter)
testdf['ingredients'] = testdf['ingredients'].map(butter)

flour = lambda x: re.sub(r'\ball purpose flour\b', 'flour', x)
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

def Tostring(Ing):
    i = ""
    for n in Ing:
        i = i + n + " , "
    return i

traindf['ingredients'] = traindf['ingredients'].apply(lambda x: Tostring(x)) 
testdf['ingredients'] = testdf['ingredients'].apply(lambda x: Tostring(x))

traindf1=traindf.loc[0:len(traindf)*0.98]
traindf2=traindf.loc[len(traindf)*0.98+1:]

Corpus1 = traindf1['ingredients']
vectorizer = TfidfVectorizer(stop_words='english',ngram_range = (1,1),analyzer="word",max_df =0.6, token_pattern=r'\w+')
tfidf1=vectorizer.fit_transform(Corpus1).toarray()

Corpus2 = traindf2['ingredients']
tfidf2=vectorizer.transform(Corpus2).toarray()

parameter = {'C':[1, 10,100]}
clf = grid_search.GridSearchCV(LR(),param_grid=parameter)
clf.fit(tfidf1, traindf1['cuisine'])
traindf2['cuisine_predicted'] = clf.predict(tfidf2)

True_Cuisine = traindf2.cuisine
Pred_Cuisine = traindf2.cuisine_predicted
accuracy_score(True_Cuisine, Pred_Cuisine, normalize=True)

Corpus3 = testdf['ingredients']
tfidf3=vectorizer.transform(Corpus3).toarray()
testdf['cuisine'] = clf.predict(tfidf3)
testdf[['id' , 'cuisine' ]].to_csv("submission_LR.csv",index=False)
Script4 = pd.read_csv("submission_LR.csv")
Script4
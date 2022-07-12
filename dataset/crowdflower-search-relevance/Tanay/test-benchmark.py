# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 17:39:57 2015

@author: tanay
"""
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn import pipeline, metrics, grid_search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import pandas as pd
#from sklearn.linear_model import LogisticRegression
from sklearn import  pipeline#, metrics, grid_search,decomposition,
from nltk.stem.porter import PorterStemmer
import re
from bs4 import BeautifulSoup
import string
from sklearn.feature_extraction import text


# array declarations
sw=[]
s_data = []
s_labels = []
t_data = []
t_labels = []
stemmer = PorterStemmer()
#stopwords tweak - more overhead
stop_words = ['http','www','img','border','color','style','padding','table','font','thi','inch','ha','width','height',
'0','1','2','3','4','5','6','7','8','9']
#stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)

#stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9']
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)

punct = string.punctuation
punct_re = re.compile('[{}]'.format(re.escape(punct)))

#remove html, remove non text or numeric, make query and title unique features for counts using prefix (accounted for in stopwords tweak)
stemmer = PorterStemmer()


train = pd.read_csv("../input/train.csv").fillna("")
test  = pd.read_csv("../input/test.csv").fillna("")


for i in range(len(train.id)):
    s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
    s=re.sub("[^a-zA-Z0-9]"," ", s)
    s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
    s_data.append(s)
    s_labels.append(str(train["median_relevance"][i]))
for i in range(len(test.id)):
    s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()
    s=re.sub("[^a-zA-Z0-9]"," ", s)
    s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
    t_data.append(s)

clf = pipeline.Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), 
('svd', TruncatedSVD(n_components=300, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), 
('ada', AdaBoostClassifier(n_estimators=100))])


param_grid = {'ada__n_estimators': [50,75]}

model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, 
                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=5)
                                 
# Fit Grid Search Model
model.fit(s_data, s_labels)
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()

for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
#    # Get best model
#    best_model = model.best_estimator_
#    
#    # Fit model with best parameters optimized for quadratic_weighted_kappa
#    best_model.fit(s_data, s_labels)
#    preds = best_model.predict(t_data)

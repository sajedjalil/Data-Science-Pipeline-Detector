# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#Libraries import
import pandas as pd
import numpy as np
import csv as csv
import json
#Common Model Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import ensemble
from sklearn.multiclass import OneVsRestClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import re

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import rmsprop

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

import warnings
warnings.filterwarnings('ignore')

from collections import Counter



with open('../input/train.json', 'r') as f:
    train = json.load(f)
train_raw_df = pd.DataFrame(train)

with open('../input/test.json', 'r') as f:
    test = json.load(f)
test_raw_df = pd.DataFrame(test)

# substitute the matched pattern
def sub_match(pattern, sub_pattern, ingredients):
    for i in ingredients.index.values:
        for j in range(len(ingredients[i])):
            ingredients[i][j] = re.sub(pattern, sub_pattern, ingredients[i][j].strip())
            ingredients[i][j] = ingredients[i][j].strip()
    re.purge()
    return ingredients

#remove units
p0 = re.compile(r'\s*(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\s*[^a-z]')
train_raw_df['ingredients'] = sub_match(p0, ' ', train_raw_df['ingredients'])
# remove digits
p1 = re.compile(r'\d+')
train_raw_df['ingredients'] = sub_match(p1, ' ', train_raw_df['ingredients'])
# remove non-letter characters
p2 = re.compile('[^\w]')
train_raw_df['ingredients'] = sub_match(p2, ' ', train_raw_df['ingredients'])

#remove units
p0 = re.compile(r'\s*(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\s*[^a-z]')
test_raw_df['ingredients'] = sub_match(p0, ' ', test_raw_df['ingredients'])
# remove digits
p1 = re.compile(r'\d+')
test_raw_df['ingredients'] = sub_match(p1, ' ', test_raw_df['ingredients'])
# remove non-letter characters
p2 = re.compile('[^\w]')
test_raw_df['ingredients'] = sub_match(p2, ' ', test_raw_df['ingredients'])

y_train = train_raw_df['cuisine'].values
train_ingredients = train_raw_df['ingredients'].values
train_ingredients_update = list()
for item in train_ingredients:
    item = [x.replace(' ', '+') for x in item]
    train_ingredients_update.append(item)
X_train = [' '.join(x) for x in train_ingredients_update]

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, 
                                   min_df=2,
                                   analyzer='word',
                                   use_idf=True,
                                   sublinear_tf=True,
                                   norm='l2')
X = tfidf_vectorizer.fit_transform(X_train)

test_ingredients = test_raw_df['ingredients'].values
test_ingredients_update = list()
for item in test_ingredients:
    item = [x.replace(' ', '+') for x in item]
    test_ingredients_update.append(item)
X_test = [' '.join(x) for x in test_ingredients_update]
X_test = tfidf_vectorizer.transform(X_test)

lb = preprocessing.LabelEncoder()
lb.fit(y_train)
y = lb.transform(y_train)

svm_clf = SVC(C=100, 
                 kernel='rbf',
                 degree=3,
                 gamma=1, 
                 coef0=1,
                 shrinking=True, 
                 tol=0.001, 
                 cache_size=200, 
                 class_weight=None, 
                 verbose=False,
                 max_iter=-1,
                 decision_function_shape=None,
                 random_state=None)
svm_ovr = OneVsRestClassifier(svm_clf)

vote_est = [
    ('clf1', LogisticRegression(random_state=2018, C=7)),
    ('clf2', svm_ovr),
    ('clf3',BernoulliNB()),
    ('clf4', RandomForestClassifier(random_state=2018, criterion = 'gini', n_estimators=100)),
    ('clf5', SGDClassifier(random_state=2018, alpha=0.00001, penalty='l2', n_iter=80)),
    ('clf6', LinearSVC())
]

#Hard Vote or majority rules
clf = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard', weights = [1,5,1,2,1,1])
clf.fit(X.toarray(), y)
# svc_score = np.average(cross_val_score(clf, X,y, scoring='accuracy'))
# print("accuracy:"+str(svc_score))

y_out = clf.predict(X_test.toarray())
y_out = lb.inverse_transform(y_out)
test_id = test_raw_df.id
sub = pd.DataFrame({'id': test_id, 'cuisine': y_out}, columns=['id', 'cuisine'])
sub.to_csv('vote_output.csv', index=False)
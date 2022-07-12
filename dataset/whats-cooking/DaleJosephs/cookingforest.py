# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:00:55 2015

@author: dsj
"""

import json
import numpy as np
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
    
def tweak_json(y):
    data = ([],[],[],[]) # [Labels], [ID], [Label_idx], Ingredients)
    for line in y:
        for k,v in line.items():
            if k == 'cuisine':
                if v not in data[0]:
                    data[0].append(v)
                data[2].append(data[0].index(v))
            elif k == 'id':
                data[1].append(v)
            elif k == 'ingredients':
                print ([wnl.lemmatize(word) for word.strip() in v])
                data[3].append(';'.join([wnl.lemmatize(word) for word in v]).strip())
    return data
    
def add_answers(answer_list, clf, data, ids):
    pred = clf.predict(data)
    answer_list.append(dict(zip(ids,pred)))
    return answer_list
    
    
with open('../input/train.json','r') as trainfile:
    raw_train = json.load(trainfile)
    
with open('../input/test.json','r') as testfile:
    raw_test = json.load(testfile)
     
train_data = tweak_json(raw_train)
test_data = tweak_json(raw_test)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vec1 = CountVectorizer(token_pattern = r"\b[\w ]+\b")
vec2 = TfidfVectorizer(token_pattern = r'\b[\w ]+\b')
X_train1 = vec1.fit_transform(train_data[3])
X_train2 = vec2.fit_transform(train_data[3])
y = np.asarray(train_data[2])


X_test1 = vec1.transform(test_data[3])
X_test2 = vec2.transform(test_data[3])
test_id = test_data[1]

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation

#clf1 = RandomForestClassifier()
#clf2 = RandomForestClassifier()
clf3 = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=600, learning_rate=1)
clf4 = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=600, learning_rate=1)
clf5 = BaggingClassifier()
clf6 = BaggingClassifier()
clf7 = GradientBoostingClassifier()
clf8 = GradientBoostingClassifier()
#classifiers = [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8]

'''
for i in range(len(classifiers)):
    if i%2 == 0:
        X = X_train2
    else:
        X = X_train1
    clf = classifiers[i]
    fit = clf.fit(X,y)
    score = clf.score(X,y)
    scores = cross_validation.cross_val_score(clf, X, y, cv=5)
    print ('Classifier #%d:\tBase fit:%f' % (i, score))
    print ('CV Scores:',scores)
    print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))'''
    
scores1 = cross_validation.cross_val_score(clf3, X_train1, y, cv=5)
scores2 = cross_validation.cross_val_score(clf4, X_train2, y, cv=5)
print ('CV Scores:',scores1)
print ("Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))
print ('TF-IDF CV scores:',scores2)
print ("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:00:55 2015

@author: dsj
"""
import json
import numpy as np
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

def lemmatize_ingredients(x):
    y = []
    for item in x:
        this = item.split()
        if len(this)>1:
            rest = this[:-1]
        else:
            rest = []
        last = this[-1]
        last = wnl.lemmatize(last.strip().lower())
        rest.append(last)
        y.append(' '.join(rest))
    return y
    
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
                lemmatized = lemmatize_ingredients(v)
                data[3].append(';'.join([word for word in lemmatized]))
    return data
    
def add_answers(answer_list, clf, data, ids):
    pred = clf.predict(data)
    answer_list.append(dict(zip(ids,pred)))
    return answer_list
    
def write_submission(idx, answers, code_list):
    fname = 'sub_'+str(idx)+'.csv'
    header = 'id,cuisine\n'
    with open(fname, 'w') as f:
        f.write(header)
        for k,v in answers.iteritems():
            row = unicode(k)+','+unicode(code_list[v])+'\n'
            f.write(row)
    return
    
with open('../input/train.json', 'r') as trainfile:
    raw_train = json.load(trainfile)
    
with open('../input/test.json', 'r') as testfile:
    raw_test = json.load(testfile)
     
train_data = tweak_json(raw_train)
test_data = tweak_json(raw_test)
all_ingredients = list(set(';'.join(test_data[3]).split(';')+';'.join(train_data[3]).split(';')))
print('foo!')
#%%

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(sublinear_tf=True, vocabulary=all_ingredients, token_pattern=r'\b[\w ]+\b', min_df=5, max_df=0.9, analyzer='word')

X_train = vect.fit_transform(train_data[3])
y = np.asarray(train_data[2])

X_test = vect.fit_transform(test_data[3])
test_id = test_data[1]
#%%

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import cross_validation

clf1 = RandomForestClassifier()
clf2 = BaggingClassifier()
clf3 = SGDClassifier(loss='hinge', penalty = 'elasticnet', n_jobs=-1)
clf4 = DecisionTreeClassifier()
clf5 = DecisionTreeClassifier(criterion='entropy')
clf6 = RidgeClassifier()
clf7 = MultinomialNB()
clf8 =PassiveAggressiveClassifier(loss='log')

classifiers = [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8]
#%%
#%%
results = []

def clf_score(clf, X, y):
    scores = cross_validation.cross_val_score(clf, X, y)
    clf.fit(X,y)
    score = clf.score(X,y)
    print ('Got scores')
    results.append((clf, score, scores, (scores.mean(), scores.std()*2)))
    #print ("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))
    
for clf in classifiers:
    clf_score(clf, X_train, y)

for i in results:
    print(i[3])
#%%
'''
answers=[]
answers = add_answers(answers, clf5, X_test1, test_id)
answers = add_answers(answers, clf6, X_test2, test_id)

write_submission(5, answers[0], train_data[0])
write_submission(6, answers[1], train_data[0])'''

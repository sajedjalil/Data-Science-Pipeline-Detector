# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 12:28:58 2016

@author: RO5079
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:14:46 2016

@author: RO5079
"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
print('start')
act_train = pd.read_csv('../input/act_train.csv')
 
people = pd.read_csv('../input/people.csv')

def preprocess_acts(data, train_set=True):
    
    # Getting rid of data feature for now
    data = data.drop(['date', 'activity_id'], axis=1)
    if(train_set):
        data = data.drop(['outcome'], axis=1)
    
    ## Split off _ from people_id
    data['people_id'] = data['people_id'].apply(lambda x: x.split('_')[1])
    data['people_id'].convert_objects(convert_numeric=True) 
    
    columns = list(data.columns)
    
    # Convert strings to ints
    for col in columns[1:]:
        data[col] = data[col].fillna('type 0')
        data[col] = data[col].apply(lambda x: x.split(' ')[1])
        data[col].convert_objects(convert_numeric=True) 
    return data

def preprocess_people(data):
    # TODO refactor this duplication
    data = data.drop(['date'], axis=1)
    data['people_id'] = data['people_id'].apply(lambda x: x.split('_')[1])
#    data['people_id'] = pd.to_numeric(data['people_id']).astype(int)
    data['people_id'].convert_objects(convert_numeric=True) 
    
    #  Values in the people df is Booleans and Strings    
    columns = list(data.columns)
    bools = columns[11:]
    strings = columns[1:11]
    
    for col in bools:
        data[col] = data[col].astype(int)      
    for col in strings:
        data[col] = data[col].fillna('type 0')
        data[col] = data[col].apply(lambda x: x.split(' ')[1])
        data[col].convert_objects(convert_numeric=True) 
    return data
    
# Preprocess each df
peeps = preprocess_people(people)
#act_train = act_train.reindex(np.random.permutation(act_train.index)).reset_index(drop=True)
actions_train = preprocess_acts(act_train)
labels = act_train['outcome']
trainlen = len(act_train)
#del people , act_train
# Merege into a unified table


from sklearn.ensemble import RandomForestClassifier , ExtraTreesClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression as lr 
from sklearn.naive_bayes import MultinomialNB as mn
from sklearn.linear_model import SGDClassifier as sgd , PassiveAggressiveClassifier as pac
import random

clf = lr(C=100000)

def batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]
n_iter = 0
models = []
#for batch in batches(range(trainlen), 500000):
#    n_iter = n_iter + 1
#    print(str(n_iter))
#    clf = ExtraTreesClassifier(n_estimators=50)
#    X_train = actions_train[batch[0]:batch[-1]+1].merge(peeps, how='left', on='people_id')
#    y_train = labels[batch[0]:batch[-1]+1]
#    clf.fit(X_train, y_train)      
#    models.append(clf)
#for batch in batches(range(trainlen), 1000000):
#    n_iter = n_iter + 1
#    print(str(n_iter))
#    clf = ExtraTreesClassifier(n_estimators=50)
#    X_train = actions_train[batch[0]:batch[-1]+1].merge(peeps, how='left', on='people_id')
#    y_train = labels[batch[0]:batch[-1]+1]
#    clf.fit(X_train, y_train)      
#    models.append(clf)
X_train = actions_train.merge(peeps, how='left', on='people_id')
y_train = labels
clf.fit(X_train, y_train)  
#for iter in range(5) :
    
#    for batch in batches(range(trainlen), 400000):
#        n_iter = n_iter + 1
#        print(str(n_iter))
#        clf = ExtraTreesClassifier()
#        X_train = actions_train[batch[0]:batch[-1]+1].merge(peeps, how='left', on='people_id')
#        y_train = labels[batch[0]:batch[-1]+1]
#        clf.fit(X_train, y_train)      
#        models.append(clf)
#        del X_train , y_train , clf
#    print('reindex')    
#    act_train = act_train.reindex(np.random.permutation(act_train.index)).reset_index(drop=True)
#    actions_train = preprocess_acts(act_train)
#    labels = act_train['outcome']

print('iter over')

#del  actions_train , X_train , y_train 


act_test = pd.read_csv('../input/act_test.csv')
test_ids = act_test['activity_id']
actions_test = preprocess_acts(act_test, train_set=False)
test = actions_test.merge(peeps, how='left', on='people_id')
del act_test , actions_test
#i = 0
#hold_preds = np.empty(shape=[len(test),])
#for clf in models :
#    i = i + 1
#    test_proba = clf.predict_proba(test)
#    test_preds = test_proba[:,1]
#    hold_preds = test_preds + hold_preds    
     
#hold_preds = np.divide(hold_preds,i)
test_proba = clf.predict_proba(test)
test_preds = test_proba[:,1]
hold_preds = test_preds   
output = pd.DataFrame({ 'activity_id' : test_ids, 'outcome': hold_preds })
output.to_csv('redhat_submit_final_allrecs.csv',index=False)
    

# Format for submission

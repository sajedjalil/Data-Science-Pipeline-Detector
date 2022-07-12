# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 15:41:08 2016

@author: SMukherjee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

act_train = pd.read_csv('../input/act_train.csv')
act_test = pd.read_csv('../input/act_test.csv')
people = pd.read_csv('../input/people.csv')

# Save the test IDs for Kaggle submission
test_ids = act_test['activity_id']

def preprocess_acts(data, train_set=True):
    
    # Getting rid of data feature for now
    data = data.drop(['date', 'activity_id'], axis=1)
    if(train_set):
        data = data.drop(['outcome'], axis=1)
    
    ## Split off _ from people_id
    data['people_id'] = data['people_id'].apply(lambda x: x.split('_')[1])
    data['people_id'] = pd.to_numeric(data['people_id']).astype(int)
    
    columns = list(data.columns)
    
    # Convert strings to ints
    for col in columns[1:]:
        data[col] = data[col].fillna('type 0')
        data[col] = data[col].apply(lambda x: x.split(' ')[1])
        data[col] = pd.to_numeric(data[col]).astype(int)
    return data

def preprocess_people(data):
    
    # TODO refactor this duplication
    data = data.drop(['date'], axis=1)
    data['people_id'] = data['people_id'].apply(lambda x: x.split('_')[1])
    data['people_id'] = pd.to_numeric(data['people_id']).astype(int)
    
    #  Values in the people df is Booleans and Strings    
    columns = list(data.columns)
    bools = columns[11:]
    strings = columns[1:11]
    
    for col in bools:
        data[col] = pd.to_numeric(data[col]).astype(int)        
    for col in strings:
        data[col] = data[col].fillna('type 0')
        data[col] = data[col].apply(lambda x: x.split(' ')[1])
        data[col] = pd.to_numeric(data[col]).astype(int)
    return data
    
    # Preprocess each df
peeps = preprocess_people(people)
actions_train = preprocess_acts(act_train)
actions_test = preprocess_acts(act_test, train_set=False)

# Merege into a unified table

# Training 
features = actions_train.merge(peeps, how='left', on='people_id')
labels = act_train['outcome']

# Testing
test = actions_test.merge(peeps, how='left', on='people_id')

# Check it out...
#features.sample(10)

## Split Training Data
from sklearn.cross_validation import train_test_split

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=num_test, random_state=23)

## Out of box random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.grid_search import GridSearchCV

clf = RandomForestClassifier(n_estimators=100,min_samples_leaf=5)
#from sklearn import svm
#clf = svm.SVC()
clf.fit(X_train, y_train)

## Training Predictions
proba = clf.predict_proba(X_test)
preds = proba[:,1]
score = roc_auc_score(y_test, preds)
print("Area under ROC {0}".format(score))

# Test Set Predictions
test_proba = clf.predict_proba(test)
test_preds = test_proba[:,1]

# Format for submission
output = pd.DataFrame({ 'activity_id' : test_ids, 'outcome': test_preds })
output.head()
output.to_csv('redhat.csv', index = False)





####---------------------------------Check for overfeat-------------------------------------
from sklearn.learning_curve import learning_curve
train_sample_size, train_scores, test_scores = learning_curve(clf,features, labels,train_sizes=np.arange(10,100,10), cv=10)
#----------------------------------------Visualization---------------------------------------------
plt.xlabel("# Training sample")
plt.ylabel("Accuracy")
plt.grid();
mean_train_scores = np.mean(train_scores, axis=1)
mean_test_scores = np.mean(test_scores, axis=1)
std_train_scores = np.std(train_scores, axis=1)
std_test_scores = np.std(test_scores, axis=1)

gap = np.abs(mean_test_scores - mean_train_scores)
g = plt.figure(1)
plt.title("Learning curves for %r\n"
             "Best test score: %0.2f - Gap: %0.2f" %
             (clf, mean_test_scores.max(), gap[-1]))
plt.plot(train_sample_size, mean_train_scores, label="Training", color="b")
plt.fill_between(train_sample_size, mean_train_scores - std_train_scores,
                 mean_train_scores + std_train_scores, alpha=0.1, color="b")
plt.plot(train_sample_size, mean_test_scores, label="Cross-validation",
         color="g")
plt.fill_between(train_sample_size, mean_test_scores - std_test_scores,
                 mean_test_scores + std_test_scores, alpha=0.1, color="g")
plt.legend(loc="lower right")
g.show()
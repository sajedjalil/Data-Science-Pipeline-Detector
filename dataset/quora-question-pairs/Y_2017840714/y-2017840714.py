'''
Goal: To determine whether two strings have the same intent.

Method:
 1. Load Data
 2. Analyze Data
 3. Train classifiers
 4. Make a prediction
'''

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

import re
import nltk
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
from nltk.tokenize import sent_tokenize, word_tokenize

from collections import Counter

# Training set
train_set = pd.read_csv('../input/train.csv')

# Testing set
test_set = pd.read_csv('../input/test.csv')

# Statistics about our data set
print('Total number of question pairs: {}'.format(len(train_set)))
print('% is_duplicate: {}'.format(round(train_set['is_duplicate'].mean() * 100, 2)))
qids = pd.Series(train_set['qid1'].tolist() + train_set['qid2'].tolist())
print('Total number of questions in training set: {}'.format(len(np.unique(qids))))
print('Number of questions with multiple appearance: {}'.format(np.sum(qids.value_counts() > 1 )))

# Setting a baseline - literal guessing based on probability
duplicateP = train_set['is_duplicate'].mean()
print('Predicted Score: ', log_loss(train_set['is_duplicate'],
np.zeros_like(train_set['is_duplicate']) + duplicateP))

# Dummy submission
dummyTest = pd.read_csv('../input/test.csv')
sub = pd.DataFrame({'testid': dummyTest['test_id'], 'is_duplicate': duplicateP})
sub.to_csv('baseline_submission.csv', index = False)


# Training questions
train_qs = pd.Series(train_set['question1'].tolist() + train_set['question2'].tolist()).astype(str)

# Word Share Feature Benchmark Model
def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R
    
train_set_wordMatch = train_set.apply(word_match_share, axis = 1, raw = True)


# TF-IDF Feature
from sklearn.feature_extraction.text import TfidfVectorizer

def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}


def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

train_set_tfidf = train_set.apply(tfidf_word_match_share, axis=1, raw=True)



# More features, current does not work
'''def ent_match(row):

    q1 = row['question1']
    q2 = row['question2']
    
    q1_pos = []
    q2_pos = []
    
    same1 = []
    same2 = []
    
    for sentence in q1:
        token = nltk.word_tokenize(sentence)
        q1_pos.append(nltk.pos_tag(token))
    
    for sentence in q2:
        token = nltk.word_tokenize(sentence)
        q2_pos.append(nltk.pos_tag(token))
    
    same1.append( ({x[0] for x in q1_pos}&{y[0] for y in q2_pos}) and ({x[1] for x in q1_pos}&{y[0] for y in q2_pos}) )
    same1.append( ({x[0] for x in q2_pos}&{y[0] for y in q1_pos}) and ({x[1] for x in q2_pos}&{y[0] for y in q1_pos}) )
    
    return (len(same1) + len(same2)) / (len(q1) + len(q2))

train_set_entMatch = train_set.apply(ent_match, axis=1, raw=True)'''


# XGBoost
import xgboost as xgb

x_train = pd.DataFrame()
x_test = pd.DataFrame()

x_train['word_match'] = train_set_wordMatch
x_train['tfidf_word_match'] = train_set_tfidf


x_test['word_match'] = test_set.apply(word_match_share, axis=1, raw=True)
x_test['tfidf_word_match'] = test_set.apply(tfidf_word_match_share, axis=1, raw=True)


y_train = train_set['is_duplicate'].values


from sklearn.cross_validation import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = test_set['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('generic_xgb.csv', index=False)









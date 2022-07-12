# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
from sklearn.cross_validation import cross_val_score, train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns



import plotly.graph_objs as go
import plotly.tools as tls

print(check_output(["ls", "../input"]).decode("utf8"))

df_train = pd.read_csv("../input/train.csv")
df_train.head()

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)

dist_train = train_qs.apply(lambda x: len(x.split(' ')))

qmarks = np.mean(train_qs.apply(lambda x: '?' in x))
math = np.mean(train_qs.apply(lambda x: '[math]' in x))
fullstop = np.mean(train_qs.apply(lambda x: '.' in x))
capital_first = np.mean(train_qs.apply(lambda x: x[0].isupper()))
capitals = np.mean(train_qs.apply(lambda x: max([y.isupper() for y in x])))
numbers = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x])))

from collections import Counter

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=1):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}
from nltk.corpus import stopwords

stops = ["for"]

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
train_word_match = df_train.apply(word_match_share, axis=1, raw=True)
tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)


x_train = pd.DataFrame()
x_train['word_match'] = train_word_match
x_train['tfidf_word_match'] = tfidf_train_word_match

y_train = df_train['is_duplicate'].values

x_train['id'] = df_train['id']
x_train['is_duplicate'] = y_train

df_test = pd.read_csv("../input/test.csv")



test_word_match = df_test.apply(word_match_share, axis=1, raw=True)
tfidf_test_word_match = df_test.apply(tfidf_word_match_share, axis=1, raw=True)
    
from sklearn.metrics import roc_auc_score

x_test = pd.DataFrame()
x_test['word_match'] = test_word_match
x_test['tfidf_word_match'] = tfidf_test_word_match

#y_test = df_test['is_duplicate'].values

x_test['id'] = df_train['id']

predictors=['word_match','tfidf_word_match']
target='is_duplicate'

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import sklearn.svm as svm

X_train= x_train[predictors]
y_train = x_train[target]
X_test=x_test[predictors]
#y_test = x_test[target]

#X=np.array(X).reshape(-1,1)
#y=np.array(y).reshape(-1,1)
#X= X.tolist()
#y= y.tolist()

from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def calculate_logloss(y_true, y_pred):
    loss_cal = log_loss(y_true, y_pred)
    return loss_cal





clf= RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=100, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, bootstrap=True, oob_score=True, n_jobs=1, random_state=1, verbose=0, warm_start=False, class_weight=None)
clf.fit(X_train,y_train)
##
#accuracy=clf.score(X_test,y_test)
#print("Random Forest",accuracy)
p=clf.predict_proba(X_test)

submission=pd.DataFrame()
submission['test_id']=df_test['test_id']
submission['is_duplicate']=p[:,1]
submission.to_csv("submission.csv", index=False)

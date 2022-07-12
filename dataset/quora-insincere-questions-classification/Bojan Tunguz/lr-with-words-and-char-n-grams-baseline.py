# This kernel uses no deep learnign or embeddings
# It is based on my own old Toxic kernel https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams
# which in turn is a fork of thousand voices' kernel https://www.kaggle.com/thousandvoices/logistic-regression-with-words-and-char-n-grams
# Kernel is also based on https://www.kaggle.com/demery/character-level-tfidf-logistic-regression
# and https://www.kaggle.com/ryanzhang/tfidf-naivebayes-logreg-baseline

import numpy as np
import pandas as pd
import gc
import re
import string
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from scipy.sparse import hstack
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm

from contextlib import contextmanager
from sklearn.base import BaseEstimator, TransformerMixin

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')
test_qid = test['qid']
train_target = train['target'].values

train_text = train['question_text']
test_text = test['question_text']

del train, test
gc.collect()

all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=8000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

del word_vectorizer
gc.collect()

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char_wb',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(2, 5),
    max_features=50000)


print('\nFitting Vectorizer')
char_vectorizer.fit(all_text)

print('\nTransforming Text')
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

del train_text, test_text, char_vectorizer
gc.collect()

kf = KFold(n_splits=5, shuffle=True, random_state=43)
test_pred_char = 0
oof_pred_char = np.zeros([train_target.shape[0],])

for i, (train_index, val_index) in tqdm(enumerate(kf.split(train_target))):
    x_train, x_val = train_char_features[list(train_index)], train_char_features[list(val_index)]
    y_train, y_val = train_target[train_index], train_target[val_index]
    classifier = LogisticRegression(C=5, solver='sag')
    classifier.fit(x_train, y_train)
    val_preds = classifier.predict_proba(x_val)[:,1]
    preds = classifier.predict_proba(test_char_features)[:,1]
    test_pred_char += 0.2*preds
    oof_pred_char[val_index] = val_preds
    print(f1_score(y_val, val_preds > 0.26))




train_features = hstack((train_char_features, train_word_features))
test_features = hstack((test_char_features, test_word_features))

del train_char_features, test_char_features, train_word_features, test_word_features
gc.collect()


train_features = coo_matrix(train_features)
test_features = coo_matrix(test_features)

train_features = train_features.tocsr()
test_features = test_features.tocsr()

kf = KFold(n_splits=5, shuffle=True, random_state=43)
test_pred_tf = 0
oof_pred_tf = np.zeros([train_target.shape[0],])

for i, (train_index, val_index) in tqdm(enumerate(kf.split(train_target))):
    x_train, x_val = train_features[list(train_index)], train_features[list(val_index)]
    y_train, y_val = train_target[train_index], train_target[val_index]
    classifier = LogisticRegression(C=3, solver='sag')
    classifier.fit(x_train, y_train)
    val_preds = classifier.predict_proba(x_val)[:,1]
    preds = classifier.predict_proba(test_features)[:,1]
    test_pred_tf += 0.2*preds
    oof_pred_tf[val_index] = val_preds
    print(f1_score(y_val, val_preds > 0.26))
    
    
kf = KFold(n_splits=5, shuffle=True, random_state=43)
test_pred_tf_2 = 0
oof_pred_tf_2 = np.zeros([train_target.shape[0],])

for i, (train_index, val_index) in tqdm(enumerate(kf.split(train_target))):
    x_train, x_val = train_features[list(train_index)], train_features[list(val_index)]
    y_train, y_val = train_target[train_index], train_target[val_index]
    classifier = LogisticRegression(C=1, solver='sag')
    classifier.fit(x_train, y_train)
    val_preds = classifier.predict_proba(x_val)[:,1]
    preds = classifier.predict_proba(test_features)[:,1]
    test_pred_tf_2 += 0.2*preds
    oof_pred_tf_2[val_index] = val_preds
    print(f1_score(y_val, val_preds > 0.26))
    
    
# oof_pred = 0.61*oof_pred_char+0.39*(0.5*oof_pred_tf + 0.5*oof_pred_tf_2)
# test_pred = 0.61*test_pred_char+0.39*(0.5*test_pred_tf + 0.5*test_pred_tf_2)

TOKENIZER = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

@contextmanager
def timer(task_name="timer"):
    # a timer cm from https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    print("----{} started".format(task_name))
    t0 = time.time()
    yield
    print("----{} done in {:.0f} seconds".format(task_name, time.time() - t0))
    
def tokenize(s):
    return TOKENIZER.sub(r' \1 ', s).split()


class NBTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1):
        self.r = None
        self.alpha = alpha

    def fit(self, X, y):
        # store smoothed log count ratio
        p = self.alpha + X[y==1].sum(0)
        q = self.alpha + X[y==0].sum(0)
        self.r = csr_matrix(np.log(
            (p / (self.alpha + (y==1).sum())) /
            (q / (self.alpha + (y==0).sum()))
        ))
        return self

    def transform(self, X, y=None):
        return X.multiply(self.r)

with timer("get Naive Bayes feature"):
    nb_transformer = NBTransformer(alpha=1).fit(train_features, train_target)
    X_nb = nb_transformer.transform(train_features)
    X_test_nb = nb_transformer.transform(test_features)
    
    
kf = KFold(n_splits=5, shuffle=True, random_state=43)
test_pred_nb = 0
oof_pred_nb = np.zeros([train_target.shape[0],])

for i, (train_index, val_index) in tqdm(enumerate(kf.split(train_target))):
    x_train, x_val = X_nb[list(train_index)], X_nb[list(val_index)]
    y_train, y_val = train_target[train_index], train_target[val_index]
    classifier = LogisticRegression(C=1, solver='sag')
    classifier.fit(x_train, y_train)
    val_preds = classifier.predict_proba(x_val)[:,1]
    preds = classifier.predict_proba(X_test_nb)[:,1]
    test_pred_nb += 0.2*preds
    oof_pred_nb[val_index] = val_preds
    print(f1_score(y_val, val_preds > 0.26))
    
del train_features, test_features
del X_nb, X_test_nb, x_train, x_val
gc.collect()

with timer("reading_data"):
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv('../input/test.csv')
    sub = pd.read_csv('../input/sample_submission.csv')
    y = train.target.values

with timer("getting ngram tfidf"):
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1,4),
        tokenizer=tokenize,
        min_df=3,
        max_df=0.9,
        strip_accents='unicode',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    ).fit(pd.concat([train['question_text'], test['question_text']]))
    X = tfidf_vectorizer.transform(train['question_text'])
    X_test = tfidf_vectorizer.transform(test['question_text'])
    
    
kf = KFold(n_splits=5, shuffle=True, random_state=43)
test_pred_big = 0
oof_pred_big = np.zeros([train_target.shape[0],])

for i, (train_index, val_index) in tqdm(enumerate(kf.split(train_target))):
    x_train, x_val = X[list(train_index)], X[list(val_index)]
    y_train, y_val = train_target[train_index], train_target[val_index]
    classifier = LogisticRegression(C=5, solver='sag')
    classifier.fit(x_train, y_train)
    val_preds = classifier.predict_proba(x_val)[:,1]
    preds = classifier.predict_proba(X_test)[:,1]
    test_pred_big += 0.2*preds
    oof_pred_big[val_index] = val_preds
    print(f1_score(y_val, val_preds > 0.26))
    
nb_transformer = NBTransformer(alpha=1).fit(X, train_target)
X_nb = nb_transformer.transform(X)
X_test_nb = nb_transformer.transform(X_test)

del X, X_test
gc.collect()
    
kf = KFold(n_splits=5, shuffle=True, random_state=43)
test_pred_nb_14 = 0
oof_pred_nb_14 = np.zeros([train_target.shape[0],])

for i, (train_index, val_index) in tqdm(enumerate(kf.split(train_target))):
    x_train, x_val = X_nb[list(train_index)], X_nb[list(val_index)]
    y_train, y_val = train_target[train_index], train_target[val_index]
    classifier = LogisticRegression(solver='lbfgs', dual=False, class_weight='balanced', C=0.69, max_iter=40)
    classifier.fit(x_train, y_train)
    val_preds = classifier.predict_proba(x_val)[:,1]
    preds = classifier.predict_proba(X_test_nb)[:,1]
    test_pred_nb_14 += 0.2*preds
    oof_pred_nb_14[val_index] = val_preds
    print(f1_score(y_val, val_preds > 0.5))
    
oof_pred = 0.63*(0.6*oof_pred_nb_14+0.4*oof_pred_big) + 0.37*(0.4*oof_pred_nb+0.51*(0.6*oof_pred_char+0.39*(0.5*oof_pred_tf + 0.5*oof_pred_tf_2)))
test_pred = 0.63*(0.6*test_pred_nb_14+0.4*test_pred_big) + 0.37*(0.4*test_pred_nb+0.51*(0.6*test_pred_char+0.39*(0.5*test_pred_tf + 0.5*test_pred_tf_2)))

np.save('oof_pred_nb_14', oof_pred_nb_14)
np.save('oof_pred_big', oof_pred_big)
np.save('oof_pred_nb', oof_pred_nb)
np.save('oof_pred_char', oof_pred_char)
np.save('oof_pred_tf', oof_pred_tf)
np.save('oof_pred_tf_2', oof_pred_tf_2)

np.save('test_pred_nb_14', test_pred_nb_14)
np.save('test_pred_big', test_pred_big)
np.save('test_pred_nb', test_pred_nb)
np.save('test_pred_char', test_pred_char)
np.save('test_pred_tf', test_pred_tf)
np.save('test_pred_tf_2', test_pred_tf_2)


score = 0
thresh = .5
for i in np.arange(0.1, 0.991, 0.01):
    temp_score = f1_score(train_target, (oof_pred > i))
    if(temp_score > score):
        score = temp_score
        thresh = i

print("CV: {}, Threshold: {}".format(score, thresh))



submission = pd.DataFrame.from_dict({'qid': test_qid})
predictions = (test_pred > thresh).astype(int)
submission['prediction'] = predictions
submission.to_csv('submission.csv', index=False)



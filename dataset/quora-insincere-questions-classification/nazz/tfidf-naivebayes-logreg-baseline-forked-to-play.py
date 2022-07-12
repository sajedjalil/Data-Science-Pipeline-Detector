# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#-------------------------------------------------------------------------------------------------------------------
# Forked from https://www.kaggle.com/ryanzhang/tfidf-naivebayes-logreg-baseline to try, test and improve if possible
#-------------------------------------------------------------------------------------------------------------------

# A adaptation of https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline
# which is then based on https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf
import os
import re
import string
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from contextlib import contextmanager
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

TOKENIZER = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
DATA_SPLIT_SEED = 42


@contextmanager
def timer(task_name="timer"):
    # a timer cm from https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    # docs available here: https://docs.python.org/3/library/contextlib.html
    print("----{} started".format(task_name))
    t0 = time.time()
    yield
    print("----{} done in {:.0f} seconds".format(task_name, time.time() - t0))


def tokenize(s):
    return TOKENIZER.sub(r' \1 ', s).split()


class NBTransformer(BaseEstimator, TransformerMixin):
    # creating a "naive Bayes transformer" on base estimator and transformer classes.
    # Docs:
    # https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    #
    # This transformer is supposed to do the following
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


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result


with timer("reading_data"):
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv('../input/test.csv')
    sub = pd.read_csv('../input/sample_submission.csv')
    y = train.target.values

with timer("getting ngram tfidf"):
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1,4),
        tokenizer=tokenize,
        min_df=10,
        max_df=0.9,
        strip_accents='unicode',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    ).fit(pd.concat([train['question_text'], test['question_text']]))
    X = tfidf_vectorizer.transform(train['question_text'])
    X_test = tfidf_vectorizer.transform(test['question_text'])
    
# getting tokens, token counts for each question and puncuation symbols statistics 
# n_tokens should be normalized to fit [0;1] as well in case we want regression to perform appropriately
ntoken_norm = 500
with timer("tokenizing_questions"):
    train['tokenized'] = train['question_text'].map(lambda x: tokenize(x))
    train['n_tokens'] = train['tokenized'].map(lambda x: len(x))
    train['n_tokens_norm'] =  train['n_tokens']/ntoken_norm
    
    test['tokenized'] = test['question_text'].map(lambda x: tokenize(x))
    test['n_tokens'] = test['tokenized'].map(lambda x: len(x))
    test['n_tokens_norm'] =  test['n_tokens']/ntoken_norm

puncts = '.?!'
with timer("counting_puncts"):
    for pnct in puncts:
        train["count_{}".format(pnct)] = train['question_text'].map(lambda x: len(x.split(pnct)) - 1)
        train["{}_per_word".format(pnct)] = train["count_{}".format(pnct)]/train['n_tokens']
        
        test["count_{}".format(pnct)] = test['question_text'].map(lambda x: len(x.split(pnct)) - 1)
        test["{}_per_word".format(pnct)] = test["count_{}".format(pnct)]/test['n_tokens']
        
    Ptrain = train[["{}_per_word".format(pnct) for pnct in puncts] + ['n_tokens_norm']].values
    Ptest = test[["{}_per_word".format(pnct) for pnct in puncts] + ['n_tokens_norm']].values


with timer("get Naive Bayes feature"):
    nb_transformer = NBTransformer(alpha=1).fit(X, y)
    # !!! should use scipy hstack option, since one of the matrices is sparse !!!
    # Another thing would be to use .tocsr() method to avoid indexing ussies further
    
    X_nb = hstack((nb_transformer.transform(X), Ptrain)).tocsr()
    X_test_nb = hstack((nb_transformer.transform(X_test), Ptest)).tocsr()
    
    # X_nb = nb_transformer.transform(X)
    # X_test_nb = nb_transformer.transform(X_test)
    
    # X_nb = X
    # X_test_nb = X_test

models = []
train_meta = np.zeros(y.shape)
test_meta = np.zeros(X_test.shape[0])
splits = list(StratifiedKFold(n_splits=20, shuffle=True, random_state=DATA_SPLIT_SEED).split(train, y))

for idx, (train_idx, valid_idx) in enumerate(splits):
    with timer("fitting on {}th split".format(idx)):
        X_train = X_nb[train_idx]
        y_train = y[train_idx]
        X_val = X_nb[valid_idx]
        y_val = y[valid_idx]
        model = LogisticRegression(solver='lbfgs', dual=False, class_weight='balanced', C=0.5, max_iter=100)
        model.fit(X_train, y_train)
        models.append(model)
        valid_pred = model.predict_proba(X_val)
        train_meta[valid_idx] = valid_pred[:,1]
        test_meta += model.predict_proba(X_test_nb)[:,1] / len(splits)

print(roc_auc_score(y, train_meta))
search_result = threshold_search(y, train_meta)
print(search_result)

sub.prediction = test_meta > search_result['threshold']
sub.to_csv("submission.csv", index=False)

# ------count "!"", "?" and use for a model input
# ------check text length in terms of tokens
# remove stop words
# 
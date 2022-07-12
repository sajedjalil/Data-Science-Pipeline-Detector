# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import gc
import gensim
print(os.listdir("../input"))

import re

from collections import defaultdict
from string import punctuation


from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# Any results you write to the current directory are saved as output.


class TfidfEmbeddingVectorizer:
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = next(iter(word2vec.values())).size
        print('Self dim', self.dim)
        self.digit = re.compile(r'(\d+)')
        
    def preproc(self, text):
        return [
            re.sub('\W+', '', t) for t in text.split() if not (t.isspace() or self.digit.search(t) or t in punctuation)
        ]

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x, tokenizer=self.preproc, stop_words='english', max_df=.95, min_df=2, binary=True)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w not in punctuation and w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
        
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)



df = pd.read_json('../input/whats-cooking-kernels-only/train.json').set_index('id')
test_df = pd.read_json('../input/whats-cooking-kernels-only/test.json').set_index('id')

traindex = df.index
testdex = test_df.index

print("Training Data Shape: ",df.shape)
print("Testing Data Shape: ", test_df.shape)
y = df.cuisine.copy()

print(y.value_counts())

df = pd.concat([df.drop("cuisine", axis=1), test_df], axis=0)
df_index = df.index
print("All Data Shape: ", df.shape)
del test_df; gc.collect();
print(df.head())


digit = re.compile(r'(\d+)')
def preproc_bigrams(text):
    return [
        t for t in text.split() if not (t.isspace() or digit.search(t) or t in '[{(<>)}]|\\@"?!.,:;\'+=*^')
    ]


# w2v_fpath = "../input/glove-twitter/glove.twitter.27B.25d.txt"
# w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_fpath, binary=False, unicode_errors='ignore')

with open("../input/fasttext-english-word-vectors-including-subwords/wiki-news-300d-1M-subword.vec", "r") as lines:
    w2v = {line.split()[0]: np.fromiter(map(float, line.split()[1:]), dtype=np.float) for line in lines}
    
vect = TfidfEmbeddingVectorizer(w2v)

# hashvect = HashingVectorizer(tokenizer=preproc_bigrams, stop_words='english', ngram_range=(1,2), n_features=5000, binary=True)

#pipeline = [
#    ('tfidf', vect),
#    ('hashing', hashvect)
#]

# pipeline_vect = FeatureUnion(pipeline)

# dummies = pipeline_vect.fit_transform(df.ingredients.str.join(' '))
dummies = vect.fit_transform(df.ingredients)
print(dummies.shape)
# df = pd.DataFrame(dummies.todense())
df = pd.DataFrame(dummies)
# print("Vocab Length: ", len(vect.get_feature_names()))
df.index = df_index

print("Number of Predictors: ", df.shape[0])
print(df.head())


X = df.loc[traindex,:]
print("Number of Cuisine Types: ", y.nunique())
print("X Shape: ",X.shape)
test_df = df.loc[testdex,:]
print("Test DF Shape: ", test_df.shape)
del df; gc.collect();


from lightgbm import LGBMClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report


params = {
    'multi_class': 'ovr',
    'solver': 'lbfgs'
}

lgbm_params = {
    'n_estimators': 250,
    'max_depth': 25,
    'learning_rate': 0.2,
    'objective': 'multiclass',
    'n_jobs': 7
}

model = LGBMClassifier(**lgbm_params)
# model = LogisticRegression(**params)
model.fit(X, y)
print(model)


y_true = y
target_names = model.classes_
y_pred = model.predict(X)
print(classification_report(y_true, y_pred, target_names=target_names))


submission = model.predict(test_df)
submission_df = pd.Series(submission, index=testdex).rename('cuisine')
submission_df.to_csv("logistic_sub.csv", index=True, header=True)
print(submission_df.head())


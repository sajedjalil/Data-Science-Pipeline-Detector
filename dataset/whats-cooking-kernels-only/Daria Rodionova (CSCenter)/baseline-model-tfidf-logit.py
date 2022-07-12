# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import gc
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


df = pd.read_json('../input/train.json').set_index('id')
test_df = pd.read_json('../input/test.json').set_index('id')

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

import re
digit = re.compile(r'(\d+)')

from string import punctuation

def preproc(text):
    return [
        re.sub('\W+', '', t) for t in text.split() if not (t.isspace() or digit.search(t) or t in punctuation)
    ]
    
def preproc_bigrams(text):
    return [
        t for t in text.split() if not (t.isspace() or digit.match(t) or t in '[{(<>)}]|\\@"?!.,:;\'+=*^')
    ]
    

from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion

tfidfvect = CountVectorizer(tokenizer=preproc, stop_words='english', max_df=.95, min_df=2, binary=True)
hashvect = HashingVectorizer(tokenizer=preproc_bigrams, stop_words='english', ngram_range=(2,2), n_features=5000, binary=True)

pipeline = [
    ('tfidf', tfidfvect),
    ('hashing', hashvect)
]

pipeline_vect = FeatureUnion(pipeline)

dummies = pipeline_vect.fit_transform(df.ingredients.str.join(' '))
# df = pd.DataFrame(dummies.todense(),columns=pipeline_vect.get_feature_names())
df = pd.DataFrame(dummies.todense())
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
from xgboost import XGBClassifier

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
    'learning_rate': 0.2,
    'objective': 'multiclass',
    'reg_alpha': .1,
    'reg_lambda': .15,
    'n_jobs': 7
}

xgb_params = {
    'n_estimators': 53,
    'max_depth': 15,
    'learning_rate': 0.15,
    'objective': 'multi:softmax',
    'n_jobs': 7
}

# model = LGBMClassifier(**lgbm_params)
# model = LogisticRegression(**params)
model = XGBClassifier(**xgb_params)
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


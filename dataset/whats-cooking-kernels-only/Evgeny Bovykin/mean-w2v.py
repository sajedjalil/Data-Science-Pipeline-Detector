# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import gc
import re

from nltk import word_tokenize
from string import punctuation

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


digit = re.compile(r'(\d+)')


print(os.listdir("../input"))
print(os.listdir("../input/glove-global-vectors-for-word-representation"))

# Any results you write to the current directory are saved as output.

df = pd.read_json('../input/whats-cooking-kernels-only/train.json').set_index('id')
test_df = pd.read_json('../input/whats-cooking-kernels-only/test.json').set_index('id')

total_non_zero = 0

def read_embeddings():
    embeddings = {}
    
    with open('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt') as f:
        for line in map(lambda x: x.strip('\n'), f):
            row = line.split()
            word = row[0]
            vector = np.array([float(x) for x in row[1:]])
            embeddings[word] = vector
            
    return embeddings
    
class MeanEmbeddingsVectorizer:
    def __init__(self, tokenizer, embeddings):
        self.tokenizer = tokenizer
        self.embeddings = embeddings
        
    def transform(self, texts):
        ret = []
        
        for text in texts:
            ret.append(self.get_mean_embeddings(text))
            
        return ret
    
    def get_mean_embeddings(self, text):
        global total_non_zero
        
        tokens = self.tokenizer(text)
        vector = np.zeros((200,))
        non_zero = 0
        
        for token in tokens:
            if token in embeddings:
                vector += embeddings[token]
                non_zero += 1
                
        if non_zero == 0:
            return vector
            
        total_non_zero += 1
            
        return vector / non_zero
        
    def fit(self, texts):
        return self
        
    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

embeddings = read_embeddings()

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

def preproc(text):
    return [
        t for t in text.split(' ') \
        if not (t.isspace() or t.startswith('\'') or digit.search(t) or t in punctuation)
    ]
    

#vect = CountVectorizer(tokenizer=preproc, stop_words='english', binary=True)
vect = MeanEmbeddingsVectorizer(tokenizer=preproc, embeddings=embeddings)

dummies = vect.fit_transform(df.ingredients.str.join(' '))
df = pd.DataFrame(dummies)

print("Total non zero ", total_non_zero)

print(df.head())

df.index = df_index

print("Number of Predictors: ", df.shape[0])
print(df.head())


X = df.loc[traindex,:]
print("Number of Cuisine Types: ", y.nunique())
print("X Shape: ",X.shape)
test_df = df.loc[testdex,:]
print("Test DF Shape: ", test_df.shape)
del df; gc.collect();

logistic_regression_params = {
    'multi_class': 'ovr',
    'solver': 'lbfgs',
    'class_weight': 'balanced'
}

logistic_regression_cv_params = {
    'C': [0.0001, 0.1, 1]
}

random_forest_params = {
    'n_estimators': 100,
    'class_weight': 'balanced',
    'max_depth': 5
}

decision_tree_params = {
    'class_weight': 'balanced',
    'min_samples_split': 10
}

mlp_params = {
    'max_iter': 1000
}

mlp_cv_params = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'alpha': [0.0001, 0.01]
}

model = MLPClassifier(**mlp_params)

model = GridSearchCV(model, mlp_cv_params)

model.fit(X, y)
print(model)


y_true = y
target_names = model.classes_
y_pred = model.predict(X)
print(classification_report(y_true, y_pred, target_names=target_names))


submission = model.predict(test_df)
submission_df = pd.Series(submission, index=testdex).rename('cuisine')
submission_df.to_csv("w2v.csv", index=True, header=True)
print(submission_df.head())

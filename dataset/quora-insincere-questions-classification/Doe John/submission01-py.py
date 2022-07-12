# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Importing Datasets
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')

# Data preparation for the Vectorizer
train_text   = train['question_text']
test_text    = test['question_text']
all_text     = pd.concat([train_text, test_text])
train_target = train.target

# Fitting our data using a 'word' Vectorizer
tfidf = TfidfVectorizer(
        sublinear_tf = True, 
        strip_accents = 'unicode',
        analyzer = 'word',
        stop_words = 'english',
        token_pattern = r'\w{1,}',
        ngram_range = (1, 1),
        max_features = 20000)
tfidf.fit(all_text)
train_features = tfidf.transform(train_text)
test_features  = tfidf.transform(test_text)

# Training our model
clsf = LogisticRegression()
cv_score = np.mean(cross_val_score(clsf, train_features, train_target, cv = 5, scoring = 'roc_auc'))
print(cv_score*100)
clsf.fit(train_features, train_target)

y_pred = clsf.predict(test_features)

subm = test.drop(['question_text'], axis = 1)
subm['prediction'] = y_pred
subm.to_csv('./submission.csv', index = False)


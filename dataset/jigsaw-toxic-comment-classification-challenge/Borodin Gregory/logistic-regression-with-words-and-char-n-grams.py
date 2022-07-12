import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack
from pprint import pprint

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

submission = pd.DataFrame.from_dict({'id': test['id']})

param_grid = {
    'C': [1, 10, 100, 1000],
    'max_iter': [50, 100, 200],
    'solver': ['sag'],
}

for class_name in class_names:
    train_target = train[class_name]
    clf = GridSearchCV(
        estimator=LogisticRegression(),
        param_grid=param_grid,
    )
    clf.fit(train_features, train_target)
    submission[class_name] = clf.predict_proba(test_features)[:, 1]

    print()
    print('CV results: {}'.format(class_name))
    print(clf.cv_results_)
    print()

submission.to_csv('submission.csv', index=False)

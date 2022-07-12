# This script shows how to obtain a quite reasonable score (0.334) by just using
# a dozen lines of built-in scikit-learn (plus imports).
# Minimalistic models are not easy to beat in NLP!
# For a thorough discussion of the tried models, see the commented notebook:
# https://www.kaggle.com/marcospinaci/talking-plots-1-sklearn-classifiers-0-334
# N.B. I have not tried to minimize the number of lines, just thrown
# away the unused ones. By sacrificing readibility one can easily halve them.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

models = [('MultiNB', MultinomialNB(alpha=0.03)),
          ('Calibrated MultiNB', CalibratedClassifierCV(
              MultinomialNB(alpha=0.03), method='isotonic')),
          ('Calibrated BernoulliNB', CalibratedClassifierCV(
              BernoulliNB(alpha=0.03), method='isotonic')),
          ('Calibrated Huber', CalibratedClassifierCV(
              SGDClassifier(loss='modified_huber', alpha=1e-4,
                            max_iter=10000, tol=1e-4), method='sigmoid')),
          ('Logit', LogisticRegression(C=30))]

train = pd.read_csv('../input/train.csv')
vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(models, voting='soft', weights=[3,3,3,1,1])
X_train = vectorizer.fit_transform(train.text.values)
authors = ['MWS','EAP','HPL']
y_train = train.author.apply(authors.index).values
clf.fit(X_train, y_train)

test = pd.read_csv('../input/test.csv', index_col=0)
X_test = vectorizer.transform(test.text.values)
results = clf.predict_proba(X_test)
pd.DataFrame(results, index=test.index, columns=authors).to_csv('results.csv')
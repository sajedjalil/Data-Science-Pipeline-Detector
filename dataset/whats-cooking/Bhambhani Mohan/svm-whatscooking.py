# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))



import json
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


train = json.load(open('../input/train.json'))
test = json.load(open('../input/test.json'))

train_as_text = [' '.join(sample['ingredients']).lower() for sample in train]
train_cuisine = [sample['cuisine'] for sample in train]

test_as_text = [' '.join(sample['ingredients']).lower() for sample in test]



tfidf_enc = TfidfVectorizer(binary=True)
lbl_enc = LabelEncoder()

X = tfidf_enc.fit_transform(train_as_text)
# X = X.astype('float16')

X_test = tfidf_enc.transform(test_as_text)
# X_test = X_test.astype('float16')

y = lbl_enc.fit_transform(train_cuisine)
print(X.shape)
print(y.shape)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

# clf = SVC(C=100, kernel='rbf', degree=3,
          # gamma=1, coef0=1, shrinking=True, 
          # probability=False, tol=0.001, cache_size=200,
          # class_weight=None, max_iter=-1,
          # decision_function_shape=None, random_state=None)
# model = OneVsRestClassifier(clf, n_jobs=4)
print(y[0:10])
clf.fit(X.toarray()	,y)

y_test = clf.predict(X_test.toarray())
test_cuisine = lbl_enc.inverse_transform(y_test)


test_id = [sample['id'] for sample in test]

submission_df = pd.DataFrame({'id': test_id, 'cuisine': test_cuisine}, columns=['id', 'cuisine'])
submission_df.to_csv('svm_submission.csv', index=False)

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import json
from pandas import DataFrame
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer


with open('../input/train.json') as train_f, open('../input/test.json') as test_f:
    train_data = json.load(train_f)
    test_data = json.load(test_f)

train_X = [' '.join(e['ingredients']) for e in train_data]
train_Y = [e['cuisine'] for e in train_data]
test_X = [' '.join(e['ingredients']) for e in test_data]
test_id = [e['id'] for e in test_data]

le = LabelEncoder()
ngram_vectorizer = CountVectorizer()
train_Y = le.fit_transform(train_Y)
train_X = ngram_vectorizer.fit_transform(train_X).toarray()
test_X = ngram_vectorizer.transform(test_X).toarray()

rf_classifier = RandomForestClassifier()
rf_classifier.fit(train_X, train_Y)

test_Y = rf_classifier.predict(test_X)
test_Y = le.inverse_transform(test_Y)

d = DataFrame(data=OrderedDict([('id', test_id), ('cuisine', test_Y)]))
d.to_csv('submission.csv', index=False)
# Any results you write to the current directory are saved as output.
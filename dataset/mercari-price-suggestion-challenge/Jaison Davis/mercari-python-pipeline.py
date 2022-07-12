# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score

print(os.listdir("../input"))

knownData = pd.read_csv('../input/train.tsv', sep='\t')

knownData.loc[knownData['brand_name'].isnull().values,'brand_name'] = 'missingvalue'
knownData.loc[knownData['item_description'].isnull().values,'item_description'] = 'missingvalue'

knownData.loc[knownData['brand_name'].isnull().values]

knownData.loc[knownData['item_description'] == 'No description yet', 'item_description'] = 'missingvalue'

knownData.item_description = knownData.name + knownData.item_description + knownData.brand_name

from sklearn.model_selection import train_test_split

X_train, X_val = train_test_split(knownData)


text_pipeSGD = Pipeline([
                    ('vect', CountVectorizer(min_df=157, ngram_range=(1, 3), stop_words='english')),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDRegressor(loss='squared_loss', penalty='l2',
                                        alpha=1e-3, random_state=42,
                                          max_iter=40, tol=None, shuffle=True))])
                                          
X_train.loc[X_train['item_description'].isnull().values,]

text_pipeSGD.fit(X_train.item_description, X_train.price)

valPredSGD = text_pipeSGD.predict(X_val.item_description)

from sklearn.metrics import mean_squared_error
from math import sqrt

sqrt(mean_squared_error(X_val.price, valPredSGD))

output = pd.DataFrame({'test_id': X_val.train_id, 'price': valPredSGD})
output.to_csv('output.csv', index=False)


test = pd.read_csv('../input/test_stg2.tsv', sep='\t')


test.loc[test['brand_name'].isnull().values,'brand_name'] = 'missingvalue'
test.loc[test['item_description'].isnull().values,'item_description'] = 'missingvalue'

test.loc[test['brand_name'].isnull().values]

test.loc[test['item_description'] == 'No description yet', 'item_description'] = 'missingvalue'
test.item_description = test.name + test.item_description + test.brand_name
test.item_description = test.item_description.str.lower()
test["item_description"] = test['item_description'].str.replace('[^\w\s]','')
testPredSGD = text_pipeSGD.predict(test.item_description)

output = pd.DataFrame({'test_id': test.test_id, 'price': testPredSGD})
output.to_csv('output.csv', index=False)
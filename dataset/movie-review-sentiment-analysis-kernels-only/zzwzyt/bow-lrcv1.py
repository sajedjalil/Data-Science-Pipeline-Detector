# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#bag of words + LR
import re

print('read data')
train_df = pd.read_csv('../input/train.tsv', sep='\t', escapechar='\\', nrows=None)
test_df = pd.read_csv('../input/test.tsv', sep='\t', escapechar='\\', nrows=None)
samp_df = pd.read_csv('../input/sampleSubmission.csv', sep='\t', escapechar='\\', nrows=None)

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = ' '.join(words)
    return words
    
print('data transform')
train_words = list(train_df.Phrase.apply(clean_text))
test_words = list(test_df.Phrase.apply(clean_text))

from sklearn.feature_extraction.text import CountVectorizer
count_Vectorizer = CountVectorizer()
count_mtx = count_Vectorizer.fit_transform(train_words+test_words)

len_train = len(train_words)
train_count_mtx = count_mtx[:len_train]
test_count_mtx = count_mtx[len_train:]

# fit and predict
from sklearn.linear_model import LogisticRegressionCV
lrCV = LogisticRegressionCV()
print('fit start...')
lrCV.fit(train_count_mtx, train_df.Sentiment)
results = lrCV.predict(test_count_mtx)

# save file
output = pd.DataFrame({'PhraseId':test_df.PhraseId, 'Sentiment':results})
pred_file_name = 'BOW_LRCV1.csv'
output.to_csv(pred_file_name, index=False)
print('done')
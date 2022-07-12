# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample = pd.read_csv("../input/sample_submission.csv")

train['binary_target'] = (train.target > 0.5).astype(int)
tfidf_vec = TfidfVectorizer()
tfidf_data = tfidf_vec.fit_transform(train.comment_text)
lr = LogisticRegression()
lr.fit(tfidf_data, train.binary_target)
y_proba = lr.predict_proba(tfidf_vec.transform(test.comment_text))
sample.prediction = y_proba[:, 1]
sample.to_csv("submission.csv", index=False)
# Any results you write to the current directory are saved as output.
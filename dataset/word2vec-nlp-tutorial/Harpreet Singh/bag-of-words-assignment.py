# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#importing data
train = pd.read_csv('../input/labeledTrainData.tsv', sep='\t', quoting=3)
test = pd.read_csv('../input/testData.tsv', sep='\t', quoting=3)

#cleaning train data
def cleaning(raw_review):
    remove_tags = BeautifulSoup(raw_review).get_text()
    letters = re.sub("[^a-zA-Z]"," ", remove_tags)
    lower_case = letters.lower()
    words = lower_case.split()
    stopword = stopwords.words("english")
    meaningful_words = [w for w in words if not w in stopword]
    return(" ".join(meaningful_words))

total_review = len(train["review"])
clean_train_reviews = []
for i in range(0 , total_review):
    clean_train_reviews.append(cleaning(train["review"][i]))

#creating bag of words of training set
vectorizer = CountVectorizer(max_features = 10000)
train_data_feature = vectorizer.fit_transform(clean_train_reviews)

#cleaning test data
total_test_review = len(test["review"])
clean_test_reviews = []
for i in range(0 , total_test_review):
    clean_test_reviews.append(cleaning(test["review"][i]))

#bag of words for testing set
test_data_feature = vectorizer.fit_transform(clean_test_reviews)

#Logitic Regression
logreg = LogisticRegression(C=0.1)
logreg = logreg.fit(train_data_feature, train["sentiment"])
result = logreg.predict(test_data_feature)

output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Bag_of_Words_output.csv", index=False, quoting=3 )

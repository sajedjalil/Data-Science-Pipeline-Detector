# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import PorterStemmer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


training = pd.read_csv('../input/labeledTrainData.tsv', header = 0, delimiter = "\t")

testing = pd.read_csv('../input/testData.tsv', header = 0, delimiter = "\t")
#to lower case
training['review'] = training['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# Removing punctuation
training['review'] = training['review'].str.replace('[^\w\s]','')
# Stop word removal
stop = stopwords.words('english')
training['review'] = training['review'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#Stemming
st = PorterStemmer()
training['review'] = training['review'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

training['review'].head()

X=training[['review']]  # Features
y=training[['sentiment']]

X_test = testing[['review']]  # Features

words = set(nltk.corpus.words.words())

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 200) 
                             
vectorizer.fit(X['review'])
vector = vectorizer.transform(X['review'])
train_data_features=vector.toarray()

forest = RandomForestClassifier(n_estimators = 150) 
forest = forest.fit( train_data_features, y['sentiment'])

vectorizer1 = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 200) 
                             
vectorizer1.fit(X_test['review'])
vector1 = vectorizer1.transform(X_test['review'])
test_data_features=vector1.toarray()

result = forest.predict(test_data_features)
output = pd.DataFrame( data={"id":testing["id"], "sentiment":result} )

output.to_csv('final.csv', index=False, quoting=3 )
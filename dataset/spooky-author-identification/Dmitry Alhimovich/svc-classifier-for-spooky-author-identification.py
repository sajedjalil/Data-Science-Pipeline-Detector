# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.stem.snowball import SnowballStemmer
import string

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Determine sizes of train and test data
print("train rows =", train.shape[0])
print("test rows =", test.shape[0])

# apply stemming for Text column


def stemmText(row):
    source = row["text"]
    
    text_string = "".join(l for l in source if l not in string.punctuation)
    
    stemmer = SnowballStemmer("english")       
    words = text_string.split(" ")
        
    new_words = []
        
    for i in range(len(words)):
        if words[i] != "" and words[i] != " ":
            new_words.append(stemmer.stem(words[i].strip()))

    text_string = " ".join(l for l in new_words)
    
    return text_string
        

train['textStemm'] = train.apply(stemmText, axis=1, raw=False)
test['textStemm'] = test.apply(stemmText, axis=1, raw=False)

print(train['textStemm'][:10])

# use cross validation to generate train and test datasets

from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(train['textStemm'], train['author'], test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.999, stop_words='english')                
features_train_transformed = vectorizer.fit_transform(features_train)
features_test_transformed  = vectorizer.transform(features_test)

features_realTest_transformed  = vectorizer.transform(test['textStemm'])


print(len(vectorizer.get_feature_names()))
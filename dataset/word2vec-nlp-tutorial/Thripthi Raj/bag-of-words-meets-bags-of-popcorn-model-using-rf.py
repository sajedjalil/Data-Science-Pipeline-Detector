
# Loading packages

import numpy as np 
import pandas as pd 
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


import os

# Load train data

train_df1 = pd.read_csv('../input/labeledTrainData.tsv', delimiter="\t")

train_df1.head()


# Load testdata
test_df1 = pd.read_csv('../input/testData.tsv', delimiter="\t")

test_df1.head()

# Data cleaning
train_df1_v2 = train_df1.drop(['id'], axis=1)

stopwd = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()

def clean_reviewtext(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stopwd]
    text = " ".join(text)
    return text

total_reviews = train_df1_v2["review"].size
train_reviews_cleaned = []

for i in range( 0, total_reviews):
        train_reviews_cleaned.append( clean_reviewtext( train_df1_v2["review"][i]))



from sklearn.feature_extraction.text import CountVectorizer 
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 2000) 

imdb_train_data_features = vectorizer.fit_transform(train_reviews_cleaned)

# Evaluating using random forest

from sklearn.ensemble import RandomForestClassifier
R_forest = RandomForestClassifier(n_estimators = 150) 

R_forest = R_forest.fit( imdb_train_data_features, train_df1_v2["sentiment"] )

# Evaluating test data

Total_test_reviews = len(test_df1["review"])
test_reviews_cleaned = [] 

for i in range(0,Total_test_reviews):
    test_clean_review = clean_reviewtext( test_df1["review"][i] )
    test_reviews_cleaned.append( test_clean_review )


vectorizer_test = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 2000) 

imdb_test_data_features = vectorizer_test.fit_transform(test_reviews_cleaned)

imdb_test_data_features = imdb_test_data_features.toarray()

result = R_forest.predict(imdb_test_data_features)

output = pd.DataFrame( data={"id":test_df1["id"], "sentiment":result} )

output.to_csv( "submit.csv", index=False, quoting=3 )

from __future__ import division
import string
import numpy as np
import pandas as pd
import pylab as plt
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup


train = pd.read_csv('../input/train.csv', encoding='utf-8')
test = pd.read_csv('../input/test.csv', encoding='utf-8')


# showing number of features and features names for train data
print ('number of features [train]: {}'.format(len(train.columns)))
print ('features names [train]: {}'.format(','.join(train.columns)))

# showing number of features and features names for test data
print ('number of features [test]: {}'.format(len(train.columns)))
print ('features names [test]: {}'.format(','.join(train.columns)))

# difference between train and test features names
print ('difference of features: {}'.format(','.join(train.columns.diff(test.columns))))

# number of rows for train and test data
print ('num rows train data: {}'.format(len(train.values)))
print ('num rows test data: {}'.format(len(test.values)))

unique_count = lambda x: len(pd.unique(x))

# count of unique features for train data
print ('count of unique values for train data features:')
for feature in train.columns:
    print ('{}: {}'.format(feature, unique_count(train[feature])))


# count of unique features for test data
print ('count of unique values for test data features:')
for feature in test.columns:
    print ('{}: {}'.format(feature, unique_count(test[feature])))

# create corpus text
stemmer = PorterStemmer()
soupify = lambda row: BeautifulSoup(row).get_text()
stopwords = ['http', 'www', 'img', 'border', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a',
             'the']
stopwords = text.ENGLISH_STOP_WORDS.union(stopwords)
tokenize = lambda text: [
    token for token in
    word_tokenize(' '.join([soupify(t).lower() for t in text if isinstance(t, str)]))
    if token not in stopwords and token not in string.punctuation
    ]


def create_text(train, test):
    text = pd.unique(np.concatenate((train, test)))
    tokens = tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens]
    return nltk.Text(tokens)

# some text analysis
lexical_diversity = lambda tokens: len(set(tokens)) / len(tokens)


def word_analysis(feature):
    print ('analyzing feature {} ...'.format(feature))

    # analyzing query
    text = create_text(train[feature], test[feature])
    fdist = nltk.FreqDist(text)
    top_10 = fdist.most_common(10)

    print ('lexical diversity: {}'.format(lexical_diversity(text.tokens)))
    print ('collocations:')
    text.collocations()
    text.dispersion_plot([token[0] for token in top_10])

    words, count = zip(*top_10)
    index = xrange(len(top_10))

    plt.bar(index, count)
    plt.xticks(index, words)
    plt.show()


word_analysis('query')
word_analysis('product_title')
word_analysis('product_description')

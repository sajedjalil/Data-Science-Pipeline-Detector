# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
'''
    Import necessacy dependencies,
    Declare global configurations,
    Define helper functions

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
seaborn.set(style="darkgrid", font_scale=1.3)
import spacy
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
import string
import re

DEBUG = True
# load spacy English module
nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
# Stopwords of English language for text cleaning
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"])
# Symbols for text cleaning
SYMBOLS = " ".join(string.punctuation).split(" ")

# Clean the text
# https://nicschrading.com/project/Intro-to-NLP-with-spaCy/
def cleanText(text):
    # Clean newlines
    text = text.strip().replace("\n", " ").replace("\r", " ")
    # Clean start and end quotes
    text = re.sub(r'^"|"$', '', text)
    # Remove hashtags
    text = re.sub('#[A-Za-z0-9]+','',text)
    # Remove @mentions
    text = re.sub('@[A-Za-z0-9]','',text)
    # Remove punctuations
    text = re.sub(r'[^\w\s]','',text)
    # replace HTML symbols
    text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
    # remove urls
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    # covert to lowercase
    text = text.lower()
    return text

# Tokenize and lemmatize the text using spaCy
# @param sample string representing the text
# https://nicschrading.com/project/Intro-to-NLP-with-spaCy/
def tokenizeText(sample):
    # clean the text first
    sample = cleanText(sample) # not sure cleaning should be done here
    # get the tokens using spaCy
    tokens = nlp(sample)
    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]
    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")
    return tokens
tokenize = np.vectorize(tokenizeText)

# jaccard similarity
# evaluation metric on Kaggle
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

train_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
if DEBUG:
    print("Training Metadata:") 
    train_data.info()
    print("------------------")

# One row is missing text and selected text
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)
if DEBUG:
    print("Training Metadata:")  
    train_data.info()
    print("------------------")

X = np.array(train_data.iloc[ :, 2:3]) # selected_text only
# y = np.array(train_data.sentiment) # sentiments
#sentiment_class = list(list(set(y)))

if DEBUG:
    # print(f'Training set example: {X[0]},{X.shape}')
    # print(f'Testing set example: {y[0]},{y.shape}')
    print("Counts of each sentiment:")
    print(train_data['sentiment'].value_counts())
    print("-------------------")
    # seaborn.factorplot(x="sentiment", data=train_data, kind="count", size=3, aspect=1.5, palette="PuBuGn_d")
    # plt.show();

# Labels to three classes without one-hot encoding
train_data.sentiment.replace('negative', 0, inplace=True)
train_data.sentiment.replace('neutral', 1, inplace=True)
train_data.sentiment.replace('positive', 2, inplace=True)
test_data.sentiment.replace('negative', 0, inplace=True)
test_data.sentiment.replace('neutral', 1, inplace=True)
test_data.sentiment.replace('positive', 2, inplace=True)
y = np.array(train_data.sentiment)
print(y)
# Labels One-hot Encoding
# binarizer = MultiLabelBinarizer()
# binarizer = binarizer.fit([sentiment_class])
# y = binarizer.fit_transform(y)


# Checking if the labels are consistent
# if DEBUG:
#     print(binarizer.classes_)
#     print(y.shape)
#     print(f'Positive Binary Consistency check\t: {set(y[6] == y[9])}')
#     print(f'Negative Binary Consistency check\t: {set(y[1] == y[2])}')
#     print(f'Neutral Binary Consistency check\t: {set(y[0] == y[7])}')

'''
    Data Preprocessing
    Not Including Tokenization, Lemmatization
    Using Vectorizer to Transform instead
'''
# First split the data into train, test and validation
# Could be useful for cross validation
if DEBUG: print("Splitting Train/Test : 70/30 .....")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Check if all the sets have three sentiment labels
if DEBUG:
    print("Checking labels in train set\t\t: {}".format(np.unique(y_train, axis=0).shape[0] == 3))
    print("Checking labels in test set\t\t: {}".format(np.unique(y_test, axis=0).shape[0] == 3))

#Vectorize
if DEBUG: print('Vectorizing the train set .....')
count_vec = CountVectorizer(max_features=1000, analyzer='word', tokenizer=tokenize, ngram_range=(1,2))
tfid_vec = TfidfVectorizer(max_features=8200)
hash_vec = HashingVectorizer(n_features=8200)
X_train = count_vec.fit_transform(X_train.reshape(-1)).toarray()
print(X_train.shape)
X_test = count_vec.fit_transform(X_test.reshape(-1)).toarray()
print(X_test.shape)

# Check if the train set still have the right shapes
if DEBUG:
    print("Checking data set shape after vectorization\t: {}".format(X_train.shape[0] == y_train.shape[0]))

'''
    Training
'''
# logisticRegression 
if DEBUG: print("Training with Logistic Regression .....")
logisticRegression = LogisticRegression(max_iter=200, solver='lbfgs').fit(X_train, y_train)
yhat_lr = logisticRegression.predict(X_test)
probs_lr = logisticRegression.decision_function(X_test)

# svm
# support_vector_machine = LinearSVC().fit(X_train, y_train)


# yhat_svm = support_vector_machine.predict(X_test)
# probs_svm = support_vector_machine.decision_function(X_test)
if DEBUG: 
    print("Accuracy score: {}".format(accuracy_score(y_test, yhat_lr)))


'''
    Extraction
'''
coef = logisticRegression.coef_
features = count_vec.get_feature_names()
def getSelectedText(W, data, features):
    IDs = np.array(data.textID)
    texts = np.array(data.text)
    sentiments = np.array(data.sentiment)

    selected_texts = []
    for i in range(IDs.shape[0]):
        text = tokenize(texts[i])
        sentiment = sentiments[i]

        if sentiment == 1:
            selected_texts.append(texts[i])
            continue

        sel_text = []
        for ngram in text:
            if ngram in features:
                index = features.index(ngram)
                weight = W[sentiment]
                if weight[index] >= 0:
                    sel_text.append(ngram)
        selected_text = ' '.join(sel_text)
        selected_texts.append(selected_text)
    return selected_texts
                

selected_text = getSelectedText(coef, test_data, features)


# In[ ]:

if DEBUG: print("Building CSV .....")
test_data["selected_text"] = selected_text
testCols = ["textID", "selected_text"]
test_data = test_data[testCols]
if DEBUG: print("Generating file .....")
test_data.to_csv("submission.csv", index=False)


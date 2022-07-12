import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
import string
import time
import spacy
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

def replace_string(str1, str2):
    temp1 = str1
    temp2 = str2
    temp = temp1.replace(temp2, "")
    return temp

nlp = spacy.load('en')
X_train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
sample = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
X_train.dropna(inplace=True)
X_train['text'] = X_train['text'].apply(lambda x: x.lower())
test['text'] = test['text'].apply(lambda x: x.lower())


#prevent setting with copy warning by explicitly stating X_train is independent
train = X_train.copy()

pos_train = X_train[X_train['sentiment'] == 'positive']
neutral_train = X_train[X_train['sentiment'] == 'neutral']
neg_train = X_train[X_train['sentiment'] == 'negative']
X_train['non_selected'] = X_train.apply(lambda x: replace_string(x['text'],x['selected_text']), axis = 1)

def calculate_selected_text(df_row, tol = 0):
    
    tweet = df_row['text']
    sentiment = df_row['sentiment']
    
    if(sentiment == 'neutral'):
        return tweet
    
    elif(sentiment == 'positive'):
        dict_to_use = pos_words_adj # Calculate word weights using the pos_words dictionary
    elif(sentiment == 'negative'):
        dict_to_use = neg_words_adj # Calculate word weights using the neg_words dictionary
        
    words = tweet.split()
    words_len = len(words)

    subsets = [words[i:j+1] for i in range(words_len) for j in range(i,words_len)]
    score = 0
    selection_str = '' # This will be our choice
    lst = sorted(subsets, key = len) # Sort candidates by length
    doc = nlp(str(words))
    for i in range(len(subsets)):
        new_sum = 0 # Sum for the current substring
        # Calculate the sum of weights for each word in the substring
        for p in range(len(lst[i])):
            if(lst[i][p].translate(str.maketrans('','',string.punctuation)) in dict_to_use.keys()):
                select = dict_to_use[lst[i][p].translate(str.maketrans('','',string.punctuation))]
            if(lst[i][p].translate(str.maketrans('','',string.punctuation)) in non_words.keys()):
                new_sum -= non_word_adj[lst[i][p].translate(str.maketrans('','',string.punctuation))]
        if(new_sum > score + tol):
            score = new_sum
            selection_str = lst[i]
            # tol = tol*5 # Increase the tolerance a bit each time we choose a selection

    # If we didn't find good substrings, return the whole text
    if(len(selection_str) == 0):
        selection_str = words
        
    return ' '.join(selection_str)

tfidf = CountVectorizer(max_df=0.95, min_df=2,max_features =10000, stop_words='english')

X_train_cv = tfidf.fit_transform(X_train['text'])

X_pos = tfidf.transform(pos_train['text'])
X_neutral = tfidf.transform(neutral_train['text'])
X_neg = tfidf.transform(neg_train['text'])

pos_count_df = pd.DataFrame(X_pos.toarray(), columns=tfidf.get_feature_names())
neutral_count_df = pd.DataFrame(X_neutral.toarray(), columns=tfidf.get_feature_names())
neg_count_df = pd.DataFrame(X_neg.toarray(), columns=tfidf.get_feature_names())



# Create dictionaries of the words within each sentiment group, where the values are the proportions of tweets that 
# contain those words

pos_words = {}
neutral_words = {}
neg_words = {}
non_words = {}

for k in tfidf.get_feature_names():
    pos = pos_count_df[k].sum()
    neutral = neutral_count_df[k].sum()
    neg = neg_count_df[k].sum()
    pos_words[k] = pos/pos_train.shape[0]
    neutral_words[k] = neutral/neutral_train.shape[0]
    neg_words[k] = neg/neg_train.shape[0]
    
X_train_cv_2 = tfidf.fit_transform(X_train['non_selected'])
X_non = tfidf.transform(X_train['non_selected'])

non_count_df = pd.DataFrame(X_non.toarray(), columns = tfidf.get_feature_names())

for k in tfidf.get_feature_names():
    non = non_count_df[k].sum()
    non_words[k] = non/X_train['non_selected'].shape[0]

neg_words_adj = {}
pos_words_adj = {}
neutral_words_adj = {}
non_word_adj = {}

for key, value in non_words.items():
    non_word_adj[key] = non_words[key]

for key, value in neg_words.items():
    neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])
    
for key, value in pos_words.items():
    pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])
    
for key, value in neutral_words.items():
    neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])

tol = 0.001

for index, row in test.iterrows():
    
    selected_text = calculate_selected_text(row, tol)
    
    sample.loc[sample['textID'] == row['textID'], ['selected_text']] = " ".join(set(selected_text.lower().split()))
  
sample.to_csv("submission.csv", index = False)
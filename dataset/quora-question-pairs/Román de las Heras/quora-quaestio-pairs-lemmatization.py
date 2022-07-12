import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from time import time

# Silly helper functions
def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']

def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']

def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

# transform Penn tag to WordNet tag
def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None

lemmatizer = WordNetLemmatizer()

def lemmatize(string):
    if string == '':
        return string
    tokens = word_tokenize(string)
    token_pos = pos_tag(tokens)
    tokens_lemma = []
    for word, tag in token_pos:
        wn_tag = penn_to_wn(tag)
        if wn_tag is None:
            wn_tag = wn.NOUN
        lemma = lemmatizer.lemmatize(word, wn_tag)
        tokens_lemma.append(lemma)
    return ' '.join(tokens_lemma)
        
# Test lemmatize function        
string = "I am sure this is a better example"
print(string)
print(lemmatize(string)) #I be sure this be a good example

# read datasets
ds = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# print head of datasets
print(ds.head())
print(test.head())

## Replace nan with ""
ds.ix[pd.isnull(ds['question1']), 'question1'] = ""
ds.ix[pd.isnull(ds['question2']), 'question2'] = ""
test.ix[pd.isnull(test['question1']), 'question1'] = ""
test.ix[pd.isnull(test['question2']), 'question2'] = ""

## Lemmatize questions
t0 = time()
ds.question1 = ds.question1.apply(lemmatize)
t1 = time() # ~ 07.27 min
print("t1: ", t1 - t0)

t0 = time()
ds.question2 = ds.question2.apply(lemmatize)
t1 = time() # ~ 07.38 min
print("t2: ", t1 - t0)

t0 = time()
#test.question1 = test.question1.apply(lemmatize)
t1 = time() # ~ 45.46 min
print("t3: ", t1 - t0)

t0 = time()
#test.question2 = test.question2.apply(lemmatize)
t1 = time() # ~ 46.23 min
print("t4: ", t1 - t0)

## Write datasets to CSV files
ds.to_csv('train_lemma.csv', index = False)
test.to_csv('test_lemma.csv', index = False)
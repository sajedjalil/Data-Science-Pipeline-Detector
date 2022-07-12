# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

import re
import string 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import spacy

from tqdm import trange
import random
from spacy.util import compounding,minibatch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Credit to https://www.kaggle.com/nkoprowicz/a-simple-solution-using-only-word-counts
def jaccard(str1, str2): 
    # If both strings are empty
    if len(str1) == 0 and len(str2) == 0:
        return 1
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

# %% [code]
print('train shape:',train.shape)
print('test shape:',test.shape)
train.head()
train = train.to_numpy()
test.head()
test = test.to_numpy()

nlp = spacy.load("en_core_web_lg")

print("\nLoading Data!")
VALIDATION_SPLIT = int(0.70 * 27481)
validation = train[1500:2000]
train = train[:1500]
print("T: ", train.shape)
print("V: ", validation.shape)
training_sentences = []
training_vectors = []
for sentence in train:
#     if the tweet was empty ignore it
    if (type(sentence[1]) is float and str(sentence[1]) == 'nan' ):
        continue
        
    tokens = list(nlp(sentence[1]))
    st_span_s = sentence[1].find(sentence[2])
    st_span_e = 0
    if (st_span_s < 0):
        print("Error, selected text not found in sentence!")
    else:
        st_span_e = st_span_s + len(sentence[2])
    for stoken in tokens:
        assert sentence[1][stoken.idx:stoken.idx + len(stoken.text)] == stoken.text
        is_in_selected = (stoken.idx >= st_span_s) and (stoken.idx + len(stoken.text) <= st_span_e)
            
        training_sentences.append([str(stoken),sentence[0], sentence[3], is_in_selected])
        training_vectors.append(stoken.vector)
        
validation_sentences = []
validation_vectors = []
for sentence in validation:
#     if the tweet was empty ignore it
    if (type(sentence[1]) is float and str(sentence[1]) == 'nan' ):
        continue
        
    tokens = list(nlp(sentence[1]))
    st_span_s = sentence[1].find(sentence[2])
    st_span_e = 0
    if (st_span_s < 0):
        print("Error, selected text not found in sentence!")
    else:
        st_span_e = st_span_s + len(sentence[2])
    for stoken in tokens:
        assert sentence[1][stoken.idx:stoken.idx + len(stoken.text)] == stoken.text
        is_in_selected = (stoken.idx >= st_span_s) and (stoken.idx + len(stoken.text) <= st_span_e)
            
        validation_sentences.append([str(stoken),sentence[0], sentence[3], is_in_selected])
        validation_vectors.append(stoken.vector)
        
test_sentences = []
test_vectors = []
for sentence in test:
#     if the tweet was empty ignore it
    if (type(sentence[1]) is float and str(sentence[1]) == 'nan' ):
        continue
        
    tokens = list(nlp(sentence[1]))
         
    for stoken in tokens:   
        test_sentences.append([str(stoken),sentence[0], sentence[2]])
        test_vectors.append(stoken.vector)
        
training_sentences = np.asarray(training_sentences)
training_vectors = np.asarray(training_vectors)
validation_sentences = np.asarray(validation_sentences)
validation_vectors = np.asarray(validation_vectors)
test_sentences = np.asarray(test_sentences)
test_vectors = np.asarray(test_vectors)
        
print("TS: ", training_sentences.shape)
print("TV: ", training_vectors.shape)
print("VS: ", validation_sentences.shape)
print("VV: ", validation_vectors.shape)
print("TTS: ", test_sentences.shape)
print("TTV: ", test_vectors.shape)

# Turn classification strings into a categorical integer (0='Neutral' or whatever it is)
tclassifications = np.asarray(pd.factorize(training_sentences[:, 2], sort=True)[0].tolist())

# Add classification integers to training vectors
training_vectors = np.append(training_vectors, np.array([tclassifications]).T, 1)

# Turn classification strings into a categorical integer (0='Neutral' or whatever it is)
vclassifications = np.asarray(pd.factorize(validation_sentences[:, 2], sort=True)[0].tolist())

# Add classification integers to validation vectors
validation_vectors = np.append(validation_vectors, np.array([vclassifications]).T, 1)

# Turn classification strings into a categorical integer (0='Neutral' or whatever it is)
ttclassifications = np.asarray(pd.factorize(test_sentences[:, 2], sort=True)[0].tolist())

# Add classification integers to test vectors
test_vectors = np.append(test_vectors, np.array([ttclassifications]).T, 1)

print("\nStarting Training!")

clf = svm.SVC()
clf.fit(training_vectors, training_sentences[:, 3])

print("\nDone Training!")


# validation_sentences = np.append(validation_sentences, np.array([clf.predict(validation_vectors)]).T, 1)
# # print("Out: ", validation_sentences[:,4])

# print("\nPredicting Sentences!")

# pred_sentences = []
# for sentence in validation:
# #     Get words predictions in the sentence
#     if sentence[3] == 'neutral':
#         pred_sentences.append(sentence[1])
#     else:
#         words = validation_sentences[validation_sentences[:,1] == sentence[0]]

#         words = words[words[:,4]=='True']

#         s = ""
#         first = True
#         for wi in words:
#             s += wi[0]
#             if not first:
#                 s += " "
#             else:
#                 first = False
            
#         pred_sentences.append(s)

# print("Sentences: ", [validation[:,1]])
# print("Expected: ", [validation[:,2]])
# print("Predictions: ", np.asarray(pred_sentences))    
# print("\nDone: ", len(pred_sentences) == len(validation))
    

# print("\nResults:")
# jaccard_data = np.append(np.array([validation[:,2]]).T, np.array([pred_sentences]).T, 1)
# jaccard_results = np.apply_along_axis(lambda x: jaccard(x[0], x[1]), 1, jaccard_data)
    
# print("JACCARD SCORE (VALIDATION): ", np.mean(jaccard_results))

print("\nTesting!")

# print("ts: ", test_sentences)
# print("tv: ", test_vectors)
test_sentences = np.append(test_sentences, np.array([clf.predict(test_vectors)]).T, 1)
print(test_sentences[:,3])

print("\nWriting Submission!")

with open('submission.csv', 'w+') as f:
    f.write("textID,selected_text\n")
    for sentence in test:
    #     Get words predictions in the sentence
        s = ""
        if sentence[2] == 'neutral':
            s = sentence[1]
        else:
            words = test_sentences[test_sentences[:,1] == sentence[0]]
#             print('w: ', words)

            words = words[words[:,3]=='True']

            first = True
            for wi in words:
                s += wi[0]
                if not first:
                    s += " "
                else:
                    first = False 
                    
        if ',' in s:
            s = '"' + s + '"'
                    
        f.write(sentence[0])
        f.write(',')
        f.write(s)
        f.write('\n')
        
print("\nDone!")




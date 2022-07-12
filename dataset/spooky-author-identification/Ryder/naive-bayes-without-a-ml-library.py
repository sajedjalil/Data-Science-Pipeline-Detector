# Kaggle - Spooky Author

# Naive Bayes without an AI library

import numpy as np
import pandas as pd
import copy # to... copy things

# ==============================================================================
# Train
# ==============================================================================

train = pd.read_csv("../input/train.csv")
authors = train.author.unique()

# remove punctuation
train.text = [sentence.replace(',','') for sentence in train.text]
train.text = [sentence.replace('.','') for sentence in train.text]
train.text = [sentence.replace('"','') for sentence in train.text]
train.text = [sentence.replace(';','') for sentence in train.text]
train.text = [sentence.replace('?','') for sentence in train.text]
train.text = [sentence.replace('!','') for sentence in train.text]
train.text = [sentence.replace(':','') for sentence in train.text]

# Get Word counts per author
words = {}
for i, sentence in enumerate(train.text):
    for word in sentence.split(' '):
        if word.lower() in words:
            if train.loc[i].author in words[word.lower()]:
                words[word.lower()][train.loc[i].author] += 1
            else:
                words[word.lower()][train.loc[i].author] = 1
        else:
            words[word.lower()] = {}
            words[word.lower()][train.loc[i].author] = 1
print('[COMPLETED] - Training Data Word Counts')


#Laplace-bayes Estimator
estimator = 1 # This number is arbitrary. It basically keeps use from multiplying 0's for probabilities.

for word in words:
    for author in authors:
        if author not in words[word]:
            words[word][author] = estimator
        else:
            words[word][author] += estimator

# Covert to likeihoods for each for words from their word counts ~ P(Author | Word)
for i, word in enumerate(words):
    total = 0
    for author in authors:
        total += words[word][author]

    for author in authors:
        words[word][author] /= total
print('[COMPLETED] - Word Likelihoods')

# Covert to authors dictionary
author_tmp = {}
for x in authors:
    author_tmp[x] = 0
authors = author_tmp

# Author priors ~ P(Author)
total = 0
for a in train.author:
    total += 1
    for author in authors:
        if a == author:
            authors[author] += 1

for author in authors:
    authors[author] /= total
print('[COMPLETED] - Author Priors')

# Likelihoods -> words, Priors -> authors
print('[COMPLETED] - Bayes Model')

# ==============================================================================
# Test,
# ==============================================================================
test = pd.read_csv("../input/test.csv")

# remove punctuation to match train
test.text = [sentence.replace(',','') for sentence in test.text]
test.text = [sentence.replace('.','') for sentence in test.text]
test.text = [sentence.replace('"','') for sentence in test.text]
test.text = [sentence.replace(';','') for sentence in test.text]
test.text = [sentence.replace('?','') for sentence in test.text]
test.text = [sentence.replace('!','') for sentence in test.text]
test.text = [sentence.replace(':','') for sentence in test.text]

# Testing Data ~ P(Author | Word1, ... , Wordn)
percents = []
for i, test_sentence in enumerate(test.text):
    percent = copy.copy(authors)
    for t in test_sentence.split(' '):
        if t.lower() in words:
            for a in authors:
                percent[a] *= words[t.lower()][a]
    percents.append(percent)

# Write to output file
with open('output.csv', "w") as f:
    header = '"id","EAP","HPL","MWS"\n'
    f.write(header)
    for i, p in enumerate(percents):
        f.write('"' + str(test.loc[i].id)+ '"' + ',' + str(p['EAP']) + ',' + str(p['HPL']) + ',' + str(p['MWS']) +'\n')

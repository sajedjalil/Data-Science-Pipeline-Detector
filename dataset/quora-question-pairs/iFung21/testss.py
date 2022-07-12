# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk
text = "When will you become a gay"
sens = nltk.sent_tokenize(text)

words = []

for sen in sens:
    words.append(nltk.word_tokenize(sen)) 

tags = []


for word in words:
    tags.append(nltk.pos_tag(word))
tags = tags[0]
print(tags)


wrbs = []
    
for tag in tags:
    if tag[1] == "WRB":
        wrbs.append(tag[0])
        
print(wrbs)


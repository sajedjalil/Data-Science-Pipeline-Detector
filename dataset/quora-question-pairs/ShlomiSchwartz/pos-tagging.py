# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import codecs
import csv
import nltk 
from nltk.stem import WordNetLemmatizer


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

BASE_DIR = '../input/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'

def getPOSLinks(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    text = nltk.word_tokenize(text)
    pos = nltk.pos_tag(text)
    links = []
    link = []
    active = False
    print(pos)
    for w in pos:
        part = w[1]
        word = w[0]
        if(not active and (part[:2] == "DT" or part == "WP" or part == "VB" or part == "IN")):
            active = True
        if(active):
            link.append(wordnet_lemmatizer.lemmatize(word))
        if(active and (part == "PRP" or part[:2] == "NN" or part == "." )):
            active = False
            links.append(" ".join(link))
            link = []
    return links

with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    
    for values in reader:
        print(getPOSLinks("Is there anything I can do to be a good geologist?"))
        print("-----------")
        print(getPOSLinks("What should I do to be a great geologist?"))
        break
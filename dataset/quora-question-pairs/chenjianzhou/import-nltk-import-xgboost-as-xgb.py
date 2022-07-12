# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import xgboost as xgb
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

help()

df_train = pd.read_csv('../input/train.csv').head(10000)
df_test = pd.read_csv('../input/test.csv').head(100)

x_train = pd.DataFrame()
x_test = pd.DataFrame()

#print(df_train)
line = {}
line['question1'] = "when are you become a gay"
line['question2'] = "where are you become a gay"



def spec_word_compare(row):
#    matchObj1 = re.search( r'([0-9]+)', str(row['question1']) , re.M|re.I)
#    matchObj2 = re.search( r'([0-9]+)', str(row['question2']) , re.M|re.I)
#    if matchObj1 and matchObj2:
#        return (int(matchObj1.group(1))+0.1)/(int(matchObj2.group(1))+0.1)
#    elif str(row['question1']).lower().split()[0] == str(row['question2']).lower().split()[0]:
#        return 1
#    else:
#        return 0
    word_wrb1 = ""
    word_nn1 = ""
    word_vb1 = ""
    sens = nltk.sent_tokenize(row['question1'].lower())
    words = []
    for sen in sens:
        words.append(nltk.word_tokenize(sen)) 
    tags = []
    for word in words:
        tags.append(nltk.pos_tag(word))
    print(tags)
    for tag in tags[0]:
        if tag[1][0:3] == "WRB":
            word_wrb1 = tag[0]
        elif tag[1][0:2] == "VB":
            word_vb1 = tag[0]
        elif tag[1][0:2] == "NN":
            word_nn1 = tag[0]

    word_wrb2 = ""
    word_nn2 = ""
    word_vb2 = ""
    sens = nltk.sent_tokenize(row['question2'].lower())
    words = []
    for sen in sens:
        words.append(nltk.word_tokenize(sen)) 
    tags = []
    for word in words:
        tags.append(nltk.pos_tag(word))
    for tag in tags[0]:
        if tag[1][0:3] == "WRB":
            word_wrb2 = tag[0]
        elif tag[1][0:2] == "VB":
            word_vb2 = tag[0]
        elif tag[1][0:2] == "NN":
            word_nn2 = tag[0]
    match1 = 0
    match2 = 0
    match3 = 0
    if word_wrb1 == word_wrb2:
        match1 = 1
    if word_vb1 == word_vb2:
        match2 = 1
    if word_nn1 == word_nn2:
        match3 = 1
    print(match1+match2+match3)
    print(word_wrb1+word_vb1+word_nn1)
    print(word_wrb2+word_vb2+word_nn2)
    return (match1 + match2 + match3)/3

print(spec_word_compare(line))

#matchObj1 = re.search( r'([0-9]+)', line['question1'] , re.M|re.I)
#matchObj2 = re.search( r'([0-9]+)', line['question2'] , re.M|re.I)
#if matchObj1 and matchObj2:
#    print(matchObj1.group(1)+matchObj2.group(1))
#else:
#    print(line1.lower().split()[0] == line2.lower().split()[0])



#df_test = pd.read_csv('../input/test.csv') .head(10)

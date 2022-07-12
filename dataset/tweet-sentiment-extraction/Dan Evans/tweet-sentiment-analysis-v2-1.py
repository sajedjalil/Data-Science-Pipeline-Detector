# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
import collections

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

def jaccard(str1, str2):
    """the Jaccard score of two strings. Provided by the competition rules"""
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

# stopword set
sw = set(stopwords.words('english'))
common = 100

def filter_stopwords(tokens, sws):
    """return a sublist of tokens that are not stopwords"""
    return [x for x in tokens if not x in sws]

# EXPLANATION
#
# This is my baseline model. It is founded on two observations from my EDA (not done here):
# 1) positive and negative selected_text are usually very short; roughly two words.
# 2) neutral tweet selected_text is usually the entire tweet.
#
# So the baseline model: make a word count dictionary of the selected_text tokens (basic 
# tokenization only, on whitespace, after punctuation is removed) for positive and negative 
# tweets separately.  Then, for each test tweet, if positive or negative, get this most 
# common word in the tweet from the appropriate count dictionary and return it; for 
# neutral tweets, return the whole string.
#
# Testing this approach on the training set gives me an average Jaccard of 0.544.  The 
# score on the test set is 0.55. So that's my score to improve upon in my next iteration.
#

# read in the training data and drop the one row an empty string
train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv', header=0)
train_df.dropna(inplace=True)

# preprocessing to fix the funkiness - taken from https://www.kaggle.com/dhananjay3/investigating-html
def remove_html_char_ref(i):
    i = i.replace("&quot;", '"')
    i = i.replace("&lt;", '<')
    i = i.replace("&gt;", '>')
    i = i.replace("&amp;", '&')
    return i

train_df['text'] = train_df.apply(lambda x: remove_html_char_ref(x['text']), axis=1)
train_df['selected_text'] = train_df.apply(lambda x: remove_html_char_ref(x['selected_text']), axis=1)

train_df['text_no_punc'] = train_df.apply(lambda x: re.sub("[^\w\s]","",x['text']), axis=1)
train_df['selected_text_no_punc'] = train_df.apply(lambda x: re.sub("[^\w\s]","", x['selected_text']), axis=1)
train_df['text_tokens'] = train_df.apply(lambda x: x['text_no_punc'].lower().split(), axis=1)
train_df['selected_text_tokens'] = train_df.apply(lambda x: x['selected_text_no_punc'].lower().split(), axis=1)
train_df['text_tokens_no_stop'] = train_df.apply(lambda x: filter_stopwords(x['text_tokens'], sw), axis=1)
train_df['selected_text_tokens_no_stop'] = train_df.apply(lambda x: filter_stopwords(x['selected_text_tokens'], sw), axis=1)

pos = train_df.loc[train_df['sentiment'] == 'positive']
neg = train_df.loc[train_df['sentiment'] == 'negative']


pwc = collections.Counter()
pos.apply(lambda x: pwc.update(filter_stopwords(x['selected_text_tokens_no_stop'], sw)), axis=1)
pwc_dict = dict(pwc)

nwc = collections.Counter()
neg.apply(lambda x: nwc.update(filter_stopwords(x['selected_text_tokens_no_stop'], sw)), axis=1)
nwc_dict = dict(nwc)

def strip_punc(text):
    """return text stripped of punctuation"""
    return re.sub("[^\w\s]","", text)

def tokenize(text):
    """tokenize the passed-in text"""
    return text.lower().split()
    
def choose_selected_text(text, sentiment):
    """choose the selected text for the passed-in text and sentiment"""
    if sentiment == 'neutral':
        return text
    elif sentiment == 'positive':
        ctr = pwc
    else:
        ctr = nwc
        
    text_tokens_no_stop = filter_stopwords(tokenize(strip_punc(text)), sw)
    best_count = 0
    best_word = ''
    
    for t in text_tokens_no_stop:
        if t in ctr:
            if ctr[t] > best_count:
                best_count = ctr[t]
                best_word = t
                    
    if best_count > 0:

        return best_word
    else:
        
        return text

train_df['model_text'] = train_df.apply(lambda x: choose_selected_text(x['text'], x['sentiment']), axis=1)
train_df['jaccard'] = train_df.apply(lambda x: jaccard(x['selected_text'], x['model_text']), axis=1)

print('training jaccard score: {0:.3f}'.format(np.mean(train_df['jaccard'])))

test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv', header=0)
test_df['text'] = test_df['text'].fillna('')
test_df['selected_text'] = test_df.apply(lambda x: choose_selected_text(x['text'], x['sentiment']), axis=1)

# output the correct data for scoring
import csv

sub_df = test_df.loc[:,['textID','selected_text']]

#print(sub_df.head())

#
# pandas won't get the quoting right for this submission, so we can manually write out the 
# submission file. Taken from Chris Deotte's comment here:
# https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/142746
#

f = open('submission.csv','w')
f.write('textID,selected_text\n')
for index, row in sub_df.iterrows():
    f.write('%s,"%s"\n'%(row.textID,row.selected_text))
f.close()
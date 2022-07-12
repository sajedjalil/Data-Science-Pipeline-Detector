# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
train = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
submission = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")

# %% [code]
train['len1'] = train['text'].apply(lambda x:len(str(x).split(' ')))
train['len2'] = train['selected_text'].apply(lambda x:len(str(x).split(' ')))



train_ = train.sample(frac=0.75)
train_.reset_index(drop=True,inplace=True)
train_copy = train.reset_index(drop=True)
val_ = pd.concat([train_copy,train_],axis=0).drop_duplicates(keep=False)




train["st_words"] = train['selected_text'].map(lambda x: str(x).split())
neutral_dat = train.query("sentiment =='neutral'")
negative_dat = train.query("sentiment =='negative'")
positive_dat = train.query("sentiment =='positive'")

neutral_st = []
negative_st = []
positive_st = []

def addSt(st_words):
    negative_st = negative_st+st_words

for i in neutral_dat['st_words']:
    neutral_st+=i
for i in negative_dat['st_words']:
    negative_st+=i
for i in positive_dat['st_words']:
    positive_st+=i

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    token = {'.':'||period||',
             ',':'||comma||',
             '"':'||quotationmark||',
             ';':'||semicolon||',
             '!':'||exclamationmark||',
             '?':'||questionmark||',
             '(':'||leftparenthesis||',
             ')':'||rightparenthesis||',
             '--':'||dash||',
             '\n':'||return||'
    }
    return token

all_word=[]
def add_word(text):
    words = []
    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = str(text).replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()
    for word in text:
        if word != "" and word != " ":
            all_word.append(word)
            words.append(word)
    return words

# 创建查找表
def create_lookup_tables(all_word):
    counts = Counter(all_word)
    vocab = sorted(counts,key=counts.get,reverse=True)
    vocab_to_int = {word :i for i,word in enumerate(vocab)}
    int_to_vocab = {i :word for i,word in enumerate(vocab)}
    int_text = [vocab_to_int[word] for word in all_word]
    return vocab_to_int, int_to_vocab,int_text    

train_data = train
train_data['text_word']=train_data['text'].map(lambda x:add_word(x))

max_text_word = train_data['text_word'].map(lambda x:len(x)).max()
train_data['selected_text_word']=train_data['selected_text'].map(lambda x:add_word(x))

test['text_word'] = test['text'].map(lambda x:  add_word(x))
vocab_to_int, int_to_vocab,int_text = create_lookup_tables(all_word)

train_['selected_text_word']=train_['selected_text'].map(lambda x:add_word(x))
train_['text_word']=train_['text'].map(lambda x:add_word(x))
val_['text_word']=val_['text'].map(lambda x:add_word(x))
val_['selected_text_word']=val_['selected_text'].map(lambda x:add_word(x))

test_neutal  = test.query("sentiment=='neutral'")
test_negative = test.query("sentiment=='negative'")
test_positive = test.query("sentiment=='positive'")
test_neutal['selected_text']=test_neutal['text_word'].map(lambda x:' '.join([ word   for word in x if word in neutral_st ]))
test_negative['selected_text']=test_negative['text_word'].map(lambda x:' '.join([ word   for word in x if word in negative_st ]))
test_positive['selected_text']=test_positive['text_word'].map(lambda x:' '.join([ word   for word in x if word in positive_st]))

submission_20200419 = test_positive.append(test_neutal.append(test_negative))[['textID','selected_text']]
# %% [code]
# submission["selected_text"] = test["text"].apply(lambda x: " ".join(x.split()[-40:]))

# %% [code]
# print(os.path.exists("/kaggle/output"))
# os.mkdir("/kaggle/output")
submission_20200419.to_csv('submission.csv', index=False)
submission_20200419.head()

# %% [code]

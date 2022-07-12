# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
sample_submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

l = data[data.sentiment != 'neutral'].selected_text
l.apply( lambda x: len(x.split())).hist(bins=30)
dict_key_sent = list(l[l.apply( lambda x: len(x.split()))<3])

def prun_text(t):
    if len(t.split()) <=5 :
        return t
    #For short text, we dont prun 

    t = t.replace('?', '.')
    t = t.replace('-', '.')
    t = t.replace('!', '.')
    t_split = t.split('.')

    def max_jaccard(t, dict_key):
        return sorted([jaccard(t, key) for key in dict_key], reverse=True)[0]

    idx = np.argmax([max_jaccard(t_i, dict_key_sent) for t_i in t_split])
    return t_split[idx]

test['selected_text'] = len(test)*''
test['selected_text'][test.sentiment=='neutral'] = (test['text'][test.sentiment=='neutral'])
test['selected_text'][test.sentiment!='neutral'] = [prun_text(t) for t in test['text'][test.sentiment!='neutral']]
submission = test[['textID', 'selected_text']]
submission.to_csv('submission.csv', index=False)
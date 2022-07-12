# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
from nltk import FreqDist
import nltk
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#df = pd.read_csv('../input/test.csv')
df = open('../input/test.csv')

def clear_stopwords(context):
    letters = re.sub("[^a-zA-Z]", " ", context)
    context = letters.lower().split()
    stopword = set(stopwords.words('english'))
    clear = [c for c in context if c not in stopword]
    return clear

def remove_html(context):
    
    cleaner = re.compile('<.*?>')
    clean_text = re.sub(cleaner,'',context)
    return clean_text

def frequent(context):
    freq = FreqDist(context)
    return freq
    #return sorted(freq,key=lambda x:x[1],reverse=True)
meaning_less = ['p','would','could','via','emp','two','must','make',
                'e','c','using','r','vs','versa','based','three']
reader = csv.DictReader(df)
preds = defaultdict(list)
output = open('output.csv','w')
writer=csv.writer(output)
writer.writerow(['id','tags'])
for idx,row in enumerate(reader):
    title = clear_stopwords(row['title']) ## return list
    content = remove_html(row['content'])
    content = clear_stopwords(content)
    freq_title = frequent(title)
    freq_content = frequent(content)
    preds[row['id']].append(' '.join(title[:]))
    #writer.writerow([row['id'],' '.join(title[:3])])
    common = set(content).intersection(title)
    temp = []
    if len(common) ==0:
        for t in title:
            if t not in meaning_less:
                temp.append(t)
        #print('ID : {} , Title : {}'.format(idx+1,title))
        writer.writerow([row['id'],' '.join(temp)])
    else:
        writer.writerow([row['id'],' '.join(common)])
    #writer.writerow([row['id'],' '.join(set(content).intersection(title))])



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-
" Kaggle - Identifying the cuisine "

import pandas as pd
import numpy as np
import os as dir 
import json as js

dir.getcwd()
dir.chdir('/kaggle/input/whats-cooking-kernels-only/')

with open('train.json') as json_file:
    data=js.load(json_file)

with open('test.json') as json_file:
    data2=js.load(json_file)
    
df_train=pd.DataFrame(data)
df_train['ingredients_str']= df_train['ingredients'].apply(', '.join)

df_test=pd.DataFrame(data2)
df_test['ingredients_str']= df_test['ingredients'].apply(', '.join)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score
vect = CountVectorizer().fit(df_train['ingredients_str'])
X_train_vectorized = vect.transform(df_train['ingredients_str'])
#Uncomment the below part only for Multinomial Naive Bayes 
clfrNB = MultinomialNB(alpha = 0.1)
clfrNB = MultinomialNB(alpha = 1)
clfrNB.fit(X_train_vectorized,df_train['cuisine'] )
preds = clfrNB.predict(vect.transform(df_test['ingredients_str']))

results=pd.DataFrame({'Id':df_test['id'],'Cuisine':preds})
#write_csv(output, 'submission.csv',index=False)
results.to_csv('/kaggle/working/submission.csv',index=False)


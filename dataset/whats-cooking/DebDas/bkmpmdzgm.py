# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 21:09:33 2017

@author: debajyoti
"""
import pandas as pd
import numpy as np
import re
from pandas import DataFrame
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier 
#function to clean data
def clean_text( raw_text ,ps):#string is immutable
    list_of_word=[]
    
    for word in raw_text:
        word=re.sub("[^a-zA-Z]"," ",word)
        words_token=word.lower().split()
        str=''
        for val in words_token:
            #val=re.sub("[^a-zA-Z]", "",val)
            str=str+ps.stem(val)
        list_of_word.append(str)
    #print(list_of_word)    
    raw_text=" ".join( list_of_word )      
    return raw_text
    
dataset=pd.read_json('../input/train.json')
ps = PorterStemmer()
ingredient=dataset['ingredients']
sizeofingredients=ingredient.size
list_of_ingredient=[]
for i in range(0,sizeofingredients):
    list_of_ingredient.append(clean_text(dataset['ingredients'][i],ps))

test_dataset=pd.read_json('../input/test.json')
sizeofingredients=test_dataset['ingredients'].size
test_ingredients=[]
for i in range(0,sizeofingredients):
    test_ingredients.append(clean_text(test_dataset['ingredients'][i],ps))
  

cv = CountVectorizer(max_features = 2000)#max_features = 2000
X = cv.fit_transform(list_of_ingredient).toarray()     
X_test = cv.transform(test_ingredients).toarray() #don't use fit transform

Y=dataset['cuisine']
le = LabelEncoder()
Y = le.fit_transform(Y)
#clf=RandomForestClassifier()#(n_estimators=20,verbose=1,n_jobs=-1,min_samples_split=2,random_state=0) 
from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB(alpha=0)
clf.fit(X,Y)

 
Y_pred=clf.predict(X_test) 
Y_pred = le.inverse_transform(Y_pred)

cuisine_id=list(test_dataset['id'])
Y_ans=list(Y_pred)
d = DataFrame(data=OrderedDict([('id', cuisine_id), ('cuisine', Y_ans)]))
d.to_csv('submissionMine.csv', index=False)
np.savetxt('result2.csv', np.r_['1,2,0',np.array(cuisine_id),np.array(Y_ans)], delimiter=',', header = 'id,cuisine', comments = '', fmt='%s')

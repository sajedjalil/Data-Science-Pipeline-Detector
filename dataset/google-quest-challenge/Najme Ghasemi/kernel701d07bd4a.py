#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 23:29:34 2019

@author: hossein
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
import re
import time
#train=pd.read_csv('/home/hossein/Documents/Kaggle Competitions/google-quest-challenge/train.csv')
train = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')

#test=pd.read_csv('/home/hossein/Documents/Kaggle Competitions/google-quest-challenge/test.csv')
test = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')

#sample=pd.read_csv('/home/hossein/Documents/Kaggle Competitions/google-quest-challenge/sample_submission.csv')
sample = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')

#%%
col=test.columns
train_1=train[col]
target=train.drop(col,axis=1)
data=pd.concat([train_1,test],ignore_index=True)
m1=5000
vectorizer1 = TfidfVectorizer(stop_words='english',max_features=m1)
vect1=vectorizer1.fit_transform(data["question_title"])
vect1n=vect1.toarray()


m2=25000
vectorizer2 = TfidfVectorizer(stop_words='english',max_features=m2)
vect2=vectorizer2.fit_transform(data["question_body"])
vect2n=vect2.toarray()
print(vectorizer2.get_stop_words())

#print(vectorizer2.get_stop_words())
m3=30000
vectorizer3 = TfidfVectorizer(stop_words='english',max_features=m3)
vect3=vectorizer3.fit_transform(data["answer"])
vect3n=vect3.toarray()
#print(vectorizer3.get_stop_words())
mapcat={'TECHNOLOGY':1,'STACKOVERFLOW':2,'CULTURE':3,'SCIENCE':4,'LIFE_ARTS':5}
data.category=data.category.map(mapcat)

vect=np.zeros([6555,m1+m2+m3])
vect[:,0:m1]=vect1n
vect[:,m1:m1+m2]=vect2n
vect[:,m1+m2:]=vect3n

for j in range(data.shape[0]):
    if re.search('\.stackexchange\.',data["host"][j]):
        data["host"][j]='stackexchange'
    if re.search('\.stackoverflow\.',data["host"][j]):
        data["host"][j]='stackoverflow'
    if re.search('\.askubuntu\.',data["host"][j]):
        data["host"][j]='askubuntu'
    if re.search('\.serverfault\.',data["host"][j]):
        data["host"][j]='serverfault' 
    if re.search('\.superuser\.',data["host"][j]):
        data["host"][j]='superuser'
    time.sleep(0.05)    

#data.to_csv('/home/hossein/Documents/Kaggle Competitions/google-quest-challenge/data2.csv')        
#%%        
maphost={'stackexchange':1,'stackoverflow.com':2,'superuser.com':3,'serverfault.com':4,'askubuntu.com':5,'mathoverflow.net':6,'askubuntu':7,'superuser':8}
data.host=data.host.map(maphost)
#data.to_csv('/home/hossein/Documents/Kaggle Competitions/google-quest-challenge/data2.csv')        
#%%
vect_n=np.zeros([6555,m1+m2+m3+2])
vect_n[:,:m1+m2+m3]=vect
vect_n[:,m1+m2+m3]=np.array(data.category)
vect_n[:,m1+m2+m3+1]=np.array(data.host)

#vect_n.to_csv('/home/hossein/Documents/Kaggle Competitions/google-quest-challenge/vect_n.csv')        
#%%
vect_n_train=vect[0:6079,:]
vect_n_te=vect[6079:,:]

from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=3)
neigh.fit(vect_n_train, target)
result=neigh.predict(vect_n_te)
targetcol=target.columns
#%%
sample[targetcol]=result
#sample.to_csv('/home/hossein/Documents/Kaggle Competitions/google-quest-challenge/sample.csv')        
sample.to_csv('submission.csv', index=False)








    
        

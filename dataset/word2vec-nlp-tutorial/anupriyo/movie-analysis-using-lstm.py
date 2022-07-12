# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
train=pd.read_csv("../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip",delimiter="\t")
test=pd.read_csv("../input/word2vec-nlp-tutorial/testData.tsv.zip",delimiter="\t")
test['review'].head(2)
train.head(20)
len(train['review'])
def preprocess(reviews):
    reviews=reviews.lower()
    reviews=BeautifulSoup(reviews,"html5lib").get_text()
    reviews=re.sub('[^a-zA-Z0-9\s]','',reviews)
    reviews=reviews.split()
    reviews=[ps.stem(word) for word in reviews if not word in set(stopwords.words('english'))]
    reviewed=' '.join(reviews)
    return reviewed
train['clean_review']=train['review'].apply(lambda x:preprocess(x))
test['clean_review']=test['review'].apply(lambda x:preprocess(x))
test.head(10)
train['review'].head() 
train_text=train.clean_review.values
train_text
test_text=test.clean_review.values
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.utils import to_categorical
length=[]
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.models import Sequential
all_words=' '.join(train_text)
all_words
#all_words=word_tokenize(train_text)
dist=nltk.FreqDist(all_words)
unique_word=len(dist)
unique_word
for i in train_text:
    words=word_tokenize(i)
    l=len(words)
    length.append(l)
max_review=np.max(length)
max_words=max_review
max_features=6000
embed_dim=128
tokenizer=Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_text)
tokenizer.fit_on_texts(train_text)
x_train=tokenizer.texts_to_sequences(train_text)
x_test=tokenizer.texts_to_sequences(test_text)
x_train=pad_sequences(x_train,maxlen=max_words)
x_train
x_train.shape
x_test=pad_sequences(x_test,maxlen=max_words)
y_train=train['sentiment']
y_train=to_categorical(y_train)

y_train
y_train.shape
model=Sequential()
model.add(Embedding(max_features,embed_dim))
model.add(LSTM(64,dropout=0.04,recurrent_dropout=0.4,return_sequences=True))
model.add(LSTM(32,dropout=0.04,recurrent_dropout=0.4,return_sequences=False))
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.04))
model.add(Dense(2,activation='softmax'))
model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.01),metrics=['accuracy'])
model.summary()
model.fit(x_train,y_train,batch_size=200,epochs=3,validation_split=0.2)
pred=model.predict_classes(x_test)
y_pred=(pred>0.5)
y_pred[1]
test.iloc[1]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
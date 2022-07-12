# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train=pd.read_csv('../input/train.csv')
train.head()
train.columns
test=pd.read_csv('../input/test.csv')
test.head()
test['target']=-999
df=pd.concat([train,test])
df.columns
df.question_text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def clean(texts):
    texts=re.sub('[^a-zA-Z0-9\s]',' ',texts)
   # texts=re.sub(r'[^\w\s]','',texts)
    texts=re.sub('\s+',' ',texts)
    texts=texts.lower()
   # texts=[ps.stem(word) for word in texts if not word in set(stopwords.words('english'))]
    return texts
df['cleaned_question']=df.question_text.apply(lambda x:clean(x))
df.cleaned_question.head()
df_train=df[df['target']!=-999]
df_test=df[df['target']==-999]
df_train_x=df_train.question_text.values
df_train_y=df_train.target.values
df_test_x=df_test.question_text.values
df_test_y=df_test.target.values
from keras.utils import to_categorical
y=to_categorical(df_train_y)
from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(df_train_x,y,test_size=0.2,stratify=y)
x_train
import nltk
from nltk.tokenize import word_tokenize
length=[]
for i in x_train:
    words=word_tokenize(i)
    l=len(words)
    length.append(l)
max_review=np.max(length)
max_review
max_words=max_review
all_words=''.join(x_train)
all_words=word_tokenize(all_words)
dist=nltk.FreqDist(all_words)
unique_word=len(dist)
unique_word
max_features=unique_word
import keras
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
embed_dim=128
tokenizer=Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_train)
x_train_text=tokenizer.texts_to_sequences(x_train)
x_test=tokenizer.texts_to_sequences(df_test_x)
x_val_text=tokenizer.texts_to_sequences(x_val)
x_train_text=pad_sequences(x_train_text,maxlen=max_words)
x_test=pad_sequences(x_test,maxlen=max_words)
x_val_text=pad_sequences(x_val_text,maxlen=max_words)
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,SpatialDropout1D
model=Sequential()
model.add(Embedding(max_features,embed_dim))
model.add(LSTM(units=64,dropout=0.25,recurrent_dropout=0.25,return_sequences=True))
model.add(LSTM(32,dropout=0.25,recurrent_dropout=0.25,return_sequences=False))
model.add(Dense(2,activation='softmax'))
model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model.fit(x_train_text,y_train,validation_data=(x_val_text,y_val),batch_size=1000,epochs=1)
pred=model.predict_classes(x_test)

pred[20000]
df_test_x[20000]
# Any results you write to the current directory are saved as output.
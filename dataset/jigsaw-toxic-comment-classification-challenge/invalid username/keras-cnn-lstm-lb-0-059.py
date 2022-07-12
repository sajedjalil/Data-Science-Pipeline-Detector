# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# coding: utf-8

# In[1]:


import pandas as pd
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
import re

from keras import backend as K
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers.convolutional import Conv1D
from keras.layers import GRU, LSTM
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[30]:


def lemmatize_all(sentence):
    wnl = WordNetLemmatizer()
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag.startswith("NN"):
            yield wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            yield wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            yield wnl.lemmatize(word, pos='a')
        elif tag.startswith('R'):
            yield wnl.lemmatize(word, pos='r')
            
        else:
            yield word


def msgProcessing(raw_msg):
    m_w=[]
    words2=[]
    raw_msg=str(raw_msg)
    raw_msg = str(raw_msg.lower())
    #url_stripper= re.sub(r'Email me.*[A-Z]',"",s)
    
    #raw_msg=re.sub(r'\w*[0-9]\w*','', url_stripper)
    raw_msg=re.sub(r'[^a-zA-Z]', ' ', raw_msg)
    
    words=raw_msg.lower().split()
    #Remove words with length lesser than 3 if not w in stops
    for i in words:
        if len(i)>=2:
            words2.append(i)
    stops=set(stopwords.words('english'))
    m_w=" ".join([w for w in words2])
    return(" ".join(lemmatize_all(m_w)))


def helperFunction(df):
    print ("Data Preprocessing!!!")
    cols=['comment_text']
    df=df[cols]
    df.comment_text.replace({r'[^\x00-\x7F]+':''},regex=True,inplace=True)
    num_msg=df[cols].size
    clean_msg=[]
    for i in range(0,num_msg):
        clean_msg.append(msgProcessing(df['comment_text'][i]))
    df['Processed_msg']=clean_msg
    X=df['Processed_msg']
    print ("Data Preprocessing Ends!!!")
    return X


# In[31]:


def embedding(train,test):
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(train)
    t=len(tokenizer.word_index)+1
    trainsequences = tokenizer.texts_to_sequences(train)
    traindata = pad_sequences(trainsequences, maxlen=100)
    testsequences = tokenizer.texts_to_sequences(test)
    testdata = pad_sequences(testsequences, maxlen=100)
    return traindata, testdata,t


# In[32]:


def getTarget(y):
    ytrain=y[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    return ytrain


# In[5]:


df= pd.read_csv("../input/train.csv",encoding='latin-1')


# In[33]:


X=helperFunction(df)


# In[34]:


df2=pd.read_csv("../input/test.csv",encoding='latin-1')
df2['comment_text'].fillna('Missing',inplace=True)


# In[35]:


X2=helperFunction(df2)


# In[36]:


xtrain,xtest,vocab_size=embedding(X,X2)

# In[37]:


classes=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
ytrain=getTarget(df[classes])


# In[44]:


def buildModel(xtrain,ytrain):
    batch_size=1000
    epochs=5
    model= Sequential()
    model.add(Embedding(20000,32,input_length=100))
    model.add(Conv1D(32,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.3))
    model.add(Conv1D(64,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.35))
    model.add(Conv1D(128,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.4))
    model.add(GRU(50,return_sequences=True))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.45))
    model.add(Dense(6,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(xtrain,ytrain,batch_size=batch_size,epochs=epochs)
    model.save("toxic.h5")
    pred=model.predict(xtest)
    return pred


# In[45]:


pred= buildModel(xtrain,ytrain)




# In[48]:


def saveCSV(ytest):
    sample_submission = pd.read_csv("../input/sample_submission.csv",encoding='latin-1')
    sample_submission[classes] = ytest
    sample_submission.to_csv("toxic.csv", index=False)


# In[49]:


saveCSV(pred)
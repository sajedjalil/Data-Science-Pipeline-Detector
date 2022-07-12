# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense , LSTM , Embedding , Conv1D , Bidirectional , GRU , Dropout , MaxPool1D

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

train_X = train["question_text"].fillna("_na_").values

test_X = test["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)

test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)

test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train['target'].values



#Using Embeddings
embedding_index = dict()
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt',encoding='utf8')

for line in f:
    
    values = line.split(" ")
    words = values[0]
   
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[words]= coefs
    
f.close()
print('Loaded %s word vectors.' % len(embedding_index))

embedding_matrix = np.zeros((max_features, 300))
for word, index in tokenizer.word_index.items():
    if index > max_features - 1:
        break
    else:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector


model = Sequential()
num_filters = 64
kernel_sizes = [1,2,3,4]

model.add(Embedding(input_dim=max_features, output_dim= embed_size , input_length=maxlen,weights=[embedding_matrix], trainable=False))

model.add(Conv1D(num_filters,kernel_sizes[0],activation='elu'))
model.add(MaxPool1D(pool_size= max_len - kernel_sizes[0] +1 ))

model.add(Conv1D(num_filters,kernel_sizes[1],activation='elu'))
model.add(MaxPool1D(pool_size= max_len - kernel_sizes[1] +1 ))

model.add(Conv1D(num_filters,kernel_sizes[2],activation='elu'))
model.add(MaxPool1D(pool_size= max_len - kernel_sizes[2] +1 ))

model.add(Conv1D(num_filters,kernel_sizes[3],activation='elu'))
model.add(MaxPool1D(pool_size= max_len - kernel_sizes[3] +1 ))

model.add(Bidirectional(GRU(128,activation='relu',dropout=0.25,recurrent_dropout=0.25)))
model.add(Dropout(0.4))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(train_X,train_y,epochs=2,batch_size=1024)

pred_test_y = model.predict([test_X], batch_size=1024, verbose=1)
pred_test_y = np.where(pred_test_y>0.5,1,0)                                    #changing the threshold in this version
out_df = pd.DataFrame({"qid":test["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)
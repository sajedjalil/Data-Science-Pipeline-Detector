# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


import os
import sys
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers import LSTM, Embedding, Bidirectional, GRU
from keras.optimizers import RMSprop,Adam

def get_model(vocab_size,seq_len):
    
    model = Sequential()
  
    ##Embedding Layer
    
  
    model.add(Dense(256,input_shape = (None, 1000)))
    model.add(Dense(128))
    model.add(Dropout(0.20))
    model.add(Dense(64))
    model.add(Dropout(0.20))
    model.add(Dense(32))
    model.add(Dropout(0.20))
    model.add(Dense(20))
    model.add(Activation('softmax'))
    
    optim = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optim,
                  metrics = ['accuracy'])
    
    return(model)

train_test_df = pd.read_json('../input/train.json')
valid_df = pd.read_json('../input/test.json')


train_test_df['seperated_ingredients'] = train_test_df['ingredients'].apply(','.join)
valid_df['seperated_ingredients'] = valid_df['ingredients'].apply(','.join)

valid_df['cuisine'] = 'unknown'

all_recipes = pd.concat([train_test_df, valid_df], ignore_index=True)

train_df, test_df = train_test_split(train_test_df, test_size = 0.25)



num_labels = 20
vocab_size = 1000
batch_size = 100
 
# define Tokenizer with Vocab Size
tokenizer = Tokenizer(num_words=vocab_size, split=',')
tokenizer.fit_on_texts(all_recipes['seperated_ingredients'].values)

x_train = np.array(tokenizer.texts_to_matrix(train_df['seperated_ingredients'].values,mode='tfidf'))
x_test = np.array(tokenizer.texts_to_matrix(test_df['seperated_ingredients'].values,mode='tfidf'))

x_valid = np.array(tokenizer.texts_to_matrix(valid_df['seperated_ingredients'].values,mode='tfidf'))



le = LabelEncoder()
le.fit(train_df['cuisine'].values)
y_train = to_categorical(le.transform(train_df['cuisine'].values))
y_test = to_categorical(le.transform(test_df['cuisine'].values))

m = get_model(vocab_size, seq_len=len(x_train[0]))

#Define callbacks
check = ModelCheckpoint('./weights.hdf5', monitor='val_loss', save_best_only=True)
lr = ReduceLROnPlateau(patience = 4)


m.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test),
          batch_size = 1000, callbacks=[check,lr])

preds = m.predict(x_valid,batch_size =1000)

##undo label transform
y_hat = le.inverse_transform(np.argmax(preds, axis=1))
submit_df = pd.DataFrame()
submit_df['id'] = valid_df['id']
submit_df['cuisine'] = y_hat

submit_df.to_csv("basic_lstm_submission_new.csv", index=False)


#Define callbacks
check = ModelCheckpoint('./weights.hdf5', monitor='val_loss', save_best_only=True)
lr = ReduceLROnPlateau(patience = 4)


m.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test),
          batch_size = 1000, callbacks=[check,lr])

preds = m.predict(x_valid,batch_size =1000)
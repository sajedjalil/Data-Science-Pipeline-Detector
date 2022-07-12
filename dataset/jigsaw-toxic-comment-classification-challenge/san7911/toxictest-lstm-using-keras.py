import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
#tokenizer.fit_on_texts(list(list_sentences_train) + list(list_sentences_test))
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

inp = Input(shape=(maxlen, ))
embed_size = 128
x = Embedding(max_features, embed_size)(inp)
x = LSTM(60, return_sequences=True,name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

print("start fitting...")
model.fit(X_t,y, epochs=2, batch_size=32, validation_split=0.1)

print("start preditcting...")
y_pred = model.predict(X_te, batch_size=1024)

submission = pd.DataFrame.from_dict({'id': test['id']})
for idx, col in enumerate(list_classes):
    submission[col] = y_pred[:,idx]
submission.to_csv('submission.csv', index=False)

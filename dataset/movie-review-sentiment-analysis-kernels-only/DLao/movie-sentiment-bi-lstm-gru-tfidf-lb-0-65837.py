# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten, CuDNNLSTM, CuDNNGRU, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, SpatialDropout1D, concatenate
from keras.models import Model, Sequential, load_model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D, GlobalAveragePooling1D
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from nltk.stem import WordNetLemmatizer
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

train_f = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/train.tsv",sep="\t")
test_f = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/test.tsv",sep="\t")
corpus_sentences = list(map(str,train_f["Phrase"])) + list(map(str,test_f["Phrase"]))

def clean_text(c):
    lemmatizer = WordNetLemmatizer()
    lemmed_words = c.copy()
    i = 0
    for sentences in c:
        temp = [lemmatizer.lemmatize(j) for j in sentences.lower().split()]
        lemmed_words[i] = " ".join(temp)
        i+=1
    text = lemmed_words.copy()
    return(text)

text = clean_text(corpus_sentences)

X_train = train_f['Phrase']
y_train = train_f['Sentiment']
Xy_train = pd.concat([X_train, y_train], axis=1)

Xy_train['Phrase'] = clean_text(list(map(str,Xy_train["Phrase"])))
test_f['Phrase'] = clean_text(list(map(str,test_f["Phrase"])))

max_words = 15000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(list(text))

list_tokenized_train = tokenizer.texts_to_sequences(Xy_train["Phrase"])
list_tokenized_test = tokenizer.texts_to_sequences(test_f["Phrase"])

max_len = 80
X_train_final = pad_sequences(list_tokenized_train,maxlen=max_len)
X_test_final = pad_sequences(list_tokenized_test,maxlen=max_len)

train_dummies = pd.get_dummies(Xy_train['Sentiment'])
y_train_final = train_dummies.values

np.random.seed(226)
shuffle_indices = np.random.permutation(np.arange(len(X_train_final)))
X_trains = X_train_final[shuffle_indices]
y_trains = y_train_final[shuffle_indices]

phs = Xy_train['Phrase'][shuffle_indices]

td = 500
vec = TfidfVectorizer(max_features=td, ngram_range=(1,2))
x_tfidf = vec.fit_transform(phs).toarray()
test_tfidf = vec.transform(test_f['Phrase']).toarray()

embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))

embed_size = 300

word_index = tokenizer.word_index
nb_words = min(max_words, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_words: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    

def keras_dl(model, embed_size, batch_size, epochs):   
    inp = Input(shape = (max_len,), name = 'lstm')
    x = Embedding(max_words,embed_size,weights = [embedding_matrix], trainable = False)(inp)
    x1 = SpatialDropout1D(0.5)(x)
    
    x_lstm = Bidirectional(CuDNNLSTM(128, return_sequences = True))(x1)
    x_lstm_c1d = Conv1D(64,kernel_size=3,padding='valid',activation='tanh')(x_lstm)
    x_lstm_c1d_gp = GlobalMaxPooling1D()(x_lstm_c1d)
    
    x_gru = Bidirectional(CuDNNGRU(128, return_sequences = True))(x1)
    x_gru_c1d = Conv1D(64,kernel_size=2,padding='valid',activation='tanh')(x_gru)
    x_gru_c1d_gp = GlobalMaxPooling1D()(x_gru_c1d)

    inp2 = Input(shape = (td,), name = 'tfidf')
    x2 = BatchNormalization()(inp2)
    x2 = Dense(8, activation='tanh')(x2)
    
    x_f = concatenate([x_lstm_c1d_gp, x_gru_c1d_gp])
    x_f = BatchNormalization()(x_f)
    x_f = Dropout(0.4)(Dense(128, activation='tanh') (x_f))    
    x_f = BatchNormalization()(x_f)
    x_f = concatenate([x_f, x2])
    x_f = Dropout(0.4)(Dense(64, activation='tanh') (x_f))
    x_f = Dense(5, activation = "sigmoid")(x_f)
    model = Model(inputs = [inp, inp2], outputs = x_f)
       
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return (model)

embed_size = 300
batch_size = 256
epochs = 30
model = Sequential()

file_path = "best_model.hdf5"
check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                              save_best_only = True, mode = "min")
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

firstmodel = keras_dl(model, embed_size, batch_size, epochs)

text_model = firstmodel.fit({'lstm': X_trains, 'tfidf': x_tfidf}, y_trains, batch_size=batch_size,epochs=epochs,verbose=0,
                            validation_split = 0.1,
                            callbacks = [check_point, early_stop])

firstmodel = load_model(file_path)
pred = firstmodel.predict([np.array(X_test_final), test_tfidf], verbose = 1)
pred2 = np.round(np.argmax(pred, axis=1)).astype(int)

sub = pd.DataFrame({'PhraseId': test_f['PhraseId'],
                   'Sentiment': pred2})

sub.to_csv("DL.csv", index = False, header = True)

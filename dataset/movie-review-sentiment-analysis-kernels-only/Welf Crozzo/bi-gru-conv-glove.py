# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import copy
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import (Input, Dense, Embedding, LSTM, GRU, Bidirectional, 
                        SpatialDropout1D,  GlobalMaxPooling1D, Concatenate, 
                        Conv1D, Dropout, BatchNormalization, Activation)
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, Nadam
from keras.models import Model
from keras import backend as K

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


MAX_LEN = 100
EMBEDDING_DIM = 300
MAX_FEATURES = 100000
RANDOM_STATE = 91
GLOVE_DIR = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

def preprocessing(train, test, max_len=MAX_LEN, max_features=MAX_FEATURES, train_size=0.75):
    """
        https://www.kaggle.com/antmarakis/cnn-baseline-model
    """
    #index = []
    #for i in train.index:
    #    if len(train.loc[i, 'Phrase'])>3:
    #        index.append(i)
    #train = train.loc[index, :].reset_index()
    #train['Phrase'] = train['Phrase'].apply(lambda x: x.lower())
    X = train['Phrase'].values.tolist()
    X_test = test['Phrase'].values.tolist()
    X_tok = X + X_test
    tokenizer = Tokenizer(num_words=max_features, filters="',")
    tokenizer.fit_on_texts(X_tok)

    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=max_len)
    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, maxlen=max_len)
    
    word_index = tokenizer.word_index
    
    y = train['Sentiment'].values
        
    Y = to_categorical(y)
    X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                          Y,
                                                          train_size=train_size,
                                                          shuffle=True,
                                                          random_state=RANDOM_STATE,
                                                          stratify=y)
    #loss_weights = np.array([len(y[y==i]) for i in range(5)])
    #loss_weights = 1-loss_weights/sum(loss_weights)
    #loss_weights = (1/5+(loss_weights-1/2)/5)/2
    loss_weights = [1/5 for _ in range(5)]
    return X_train, X_valid, y_train, y_valid, X_test, loss_weights, word_index


def get_model(embedding_matrix, word_index, max_len=MAX_LEN):
    inp = Input((max_len,))
    
    x = Embedding(len(word_index) + 1,
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    input_length=max_len,
                    trainable=True)(inp)
                    
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(256, return_sequences=True))(x)
    x = Conv1D(256, 7, strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = SpatialDropout1D(0.)(x)
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = Conv1D(128, 7, strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = SpatialDropout1D(0.)(x)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = Conv1D(64, 7, strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = GlobalMaxPooling1D()(x)
    
    x = Dense(128, activation='tanh')(x)
    out = Dense(5, activation='sigmoid')(x)
    
    model = Model(inputs=inp, outputs=out)
    
    return model

def weighted_categorical_crossentropy(weights):
    """
    https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def get_glove(word_index, path=GLOVE_DIR):
    """
        https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    """
    embeddings_index = {}
    with open(path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
    
def main():
    train = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv',  sep="\t")
    test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv',  sep="\t")
    
    X_train, X_valid, y_train, y_valid, X_test, loss_weights, word_index = preprocessing(train, test)
    
    glove_emb = get_glove(word_index)
    model = get_model(glove_emb, word_index)
    model.summary()
    
    opt = Nadam(lr=1e-3, schedule_decay=0.04)
    
    model.compile(loss=weighted_categorical_crossentropy(loss_weights),
                  optimizer=opt, 
                  metrics=['accuracy'])

    mc = ModelCheckpoint('best_weights.h5',
                        monitor='val_acc', 
                        verbose=0, 
                        save_best_only=True, 
                        save_weights_only=False, 
                        mode='max', 
                        period=1)
                        
    model.fit(X_train, y_train,
              epochs = 4,
              verbose = 1,
              batch_size = 128,
              validation_data = [X_valid, y_valid],
              callbacks = [mc])
    
    X_train = X_train.tolist()+X_valid.tolist()
    
    model.load_weights('best_weights.h5')
    
    sub = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv')
    
    sub_train = model.predict(X_train, verbose=1)
    sub_test = model.predict(X_test, verbose=1)
    
    for i in range(5):
        train['prob_%i'%i] = sub_train[:][i]
        test['prob_%i'%i] = sub_test[:][i]
    sub['Sentiment'] = np.argmax(sub_test, axis=-1)
    
    sub.to_csv('sub_gru_lstm.csv', index=False)
    train.to_csv('train_gru.tsv', sep='\t')
    test.to_csv('test_gru.tsv', sep='\t')
    
if __name__=='__main__':
    main()
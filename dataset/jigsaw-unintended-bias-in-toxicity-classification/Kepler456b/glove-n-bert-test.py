# Kernels that helped me: 
# https://www.kaggle.com/thousandvoices/simple-lstm 
# https://www.kaggle.com/hiromoon166/load-bert-fine-tuning-model

# I had to break it up into 3 sections: prep, train, inference. This is third.

import numpy as np
import pandas as pd
import os
import sys
import random
import keras
import tensorflow as tf
import json

import logging

from shutil import copytree
sys.path.insert(0, '../input/pretrained-bert-including-scripts/master/bert-master')
sys.path.insert(0, '../input/kerasbert')
copytree('../input/kerasbert/keras_bert','kaggle/working')

BERT_PRETRAINED_DIR = '../input/pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12'

print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))
import tokenization  
from keras.models import load_model
from keras_bert.keras_bert.bert import get_model
from keras_bert.keras_bert.loader import load_trained_model_from_checkpoint
from keras.optimizers import Adam
adam = Adam(lr=2e-5,decay=0.01)
maxlen = 50
print('begin_build')
config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
model = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=True,seq_len=maxlen)
for i in range(110):
    model.layers[i].trainable = False

import logging
import datetime
import warnings
import pickle

from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

from keras.models import Model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda
from keras.layers import Embedding, SpatialDropout1D, Dropout, add
from keras.layers import CuDNNLSTM, GRU, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
import keras.backend as K
import re
import codecs

import zipfile

def build_marriage_model(embedding_matrix, model,num_aux_targets):
    LSTM_UNITS = 512
    DENSE_HIDDEN_UNITS = 512
    words = Input(shape=(maxlen,))
    glove_embs = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    sequence_output  = model.layers[-7].output
    bertlstm = Bidirectional(GRU(LSTM_UNITS, return_sequences=True))(sequence_output)
    glovelstm = Bidirectional(GRU(LSTM_UNITS, return_sequences=True))(glove_embs)
    conc = concatenate([bertlstm,glovelstm,])
    x = Bidirectional(GRU(LSTM_UNITS, return_sequences=True))(conc)
    x = Bidirectional(GRU(LSTM_UNITS, return_sequences=True))(x)
    hidden1 = GlobalAveragePooling1D()(x)
    hidden2 = GlobalAveragePooling1D()(bertlstm)
    hidden = concatenate([hidden1, hidden2])
    hidden = Dense(DENSE_HIDDEN_UNITS, activation='relu',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))(hidden)
    hidden = Dense(DENSE_HIDDEN_UNITS, activation='relu',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))(hidden)
    pool_output = Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),name = 'real_output')(hidden)
    #aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    print("mode inputs ", type(model.input))
    model_inputs = [model.input[0],model.input[1],model.input[2],words]
    model3  = Model(inputs=model_inputs, outputs=pool_output)#,aux_result])
    model3.compile(loss='binary_crossentropy', optimizer=adam)
    model3.summary()
    print(model_inputs)
    return model3
    
def build_simple_model(model):
    LSTM_UNITS = 512
    DENSE_HIDDEN_UNITS = 1024
    sequence_output  = model.layers[-7].output
    bertlstm = Bidirectional(GRU(256, return_sequences=False))(sequence_output)
    hidden = Dense(128, activation='relu',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))(bertlstm)
    pool_output = Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),name = 'real_output')(hidden)
    model_inputs = [model.input[0],model.input[1],model.input[2]]
    model5  = Model(inputs=model_inputs, outputs=pool_output)#,aux_result])
    adam = optimizers.adam(lr=0.01) #added line to speed up convergence
    model5.compile(loss='binary_crossentropy', optimizer=adam)
    model5.summary()
    print(model_inputs)
    return model5

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)

def convert_lines(examples, maxlen,tokenizer):
    max_seq_length = maxlen - 2
    all_tokens = []
    longer = 0
    glove_lines = []
    for example in examples:
        tokens_a = tokenizer.tokenize(example)
        if len(tokens_a)>max_seq_length: # to account for CLS and SEP minus 2
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        glove_lines.append(["[CLS]"]+tokens_a+["[SEP]"])
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"]) + [0] * (max_seq_length - len(tokens_a))
        assert len(one_token) == maxlen
        all_tokens.append(one_token)
    #print(longer)
    return np.array(all_tokens), glove_lines
    
def tokenize_glove_lines(glove_lines, embedding_index):
    token_index = 1 #index 0 reserved for padding
    embedding_matrix = [np.zeros(300)] #first item in matrix for padding
    token_to_word = {}
    word_to_token = {}
    token_to_word[0] = "<PAD>"
    word_to_token["<PAD>"] = 0
    tokenized_lines = []
    for line in glove_lines:
        tokenized_line = []
        for word in line:
            if word not in word_to_token.keys():
                embedding_array = np.zeros(300)
                word_to_token[word] = token_index
                token_to_word[token_index] = word
                token_index += 1                
                try:
                    embedding_array = embedding_index[word]
                except KeyError:
                    pass
                embedding_matrix.append(embedding_array)
            tokenized_line.append(word_to_token[word])
        tokenized_lines.append(tokenized_line)
    return tokenized_lines, embedding_matrix, token_to_word, word_to_token

def pad_tokenized_lines(tokenized_lines, maxlen):
    glove_input = []
    for line in tokenized_lines:
        line = line + [0] * (maxlen - len(line))
        assert len(line) == maxlen
        glove_input.append(np.asarray(line))
    return np.asarray(glove_input)

def save_obj(obj, name ):
    with open('../input/glovebert/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('../input/glovebert/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

embedding_matrix = load_obj("emb_file")
word_to_token = load_obj("dict_file")

#load test data
test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

test_lines = test_df['comment_text'].values
print(test_lines.shape)
print('load data done')

vocab_file = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

print("tokenizer done")

num_aux_targets = 5
embedding_matrix = np.asarray(embedding_matrix)
#model3 = build_marriage_model(embedding_matrix, model, num_aux_targets)
print("model3 done")

def tokenize_test_lines(test_lines, word_to_token):
    tokenized_lines = []
    for test_line in test_lines:
        tokenized_line = []
        for word in test_line:
            if word in word_to_token.keys():
                tokenized_line.append(word_to_token[word])
            else:
                tokenized_line.append(0)
        tokenized_lines.append(tokenized_line)
    return tokenized_lines

test_input, test_glove_lines = convert_lines(list(test_lines),maxlen,tokenizer)
test_glove_lines = tokenize_test_lines(test_glove_lines, word_to_token)
test_glove_input = pad_tokenized_lines(test_glove_lines, maxlen)

seg_input = np.zeros((test_input.shape[0],maxlen))
mask_input = np.ones((test_input.shape[0],maxlen))

BATCH_SIZE = 512

# you can load the weights by this line
print('load model')
#model3.load_weights('../input/glove16/bertglove_w5.h5')
#hehe = model3.predict([test_input, seg_input, mask_input, test_glove_input],verbose=1,batch_size=BATCH_SIZE)

model5 = build_simple_model(model)
model5.load_weights('../input/glove17/bertglove_w5.h5')
tryme = model5.predict([test_input, seg_input, mask_input],  verbose=1, batch_size=BATCH_SIZE)
print("predict finished")

submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
submission['prediction'] = tryme.flatten()
submission.reset_index(drop=False, inplace=True)
submission.to_csv('submission.csv', index=False)

#conclusion: glove and bert don't mix. simple ensemble better. 
# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import gc
import re
import operator 
import math

from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from keras.models import Model, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, Dense, add, CuDNNLSTM, CuDNNGRU, Dropout, concatenate, Bidirectional, SpatialDropout1D, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, Callback
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers


# %% [code]
# load data
train = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")
test = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")

print("Train shape : ",train.shape)
print("Test shape : ",test.shape)

df = pd.concat([train[['id','comment_text']], test], axis=0)

df['comment_text'] = df['comment_text'].apply(lambda x: x.lower())

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

df['comment_text'] = df['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"_":" ", "`":" "}

def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])    
    for p in punct:
        text = text.replace(p, f' {p} ')     
    return text

df['comment_text'] = df['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

train = df.iloc[:train.shape[0],:]
test = df.iloc[train.shape[0]:,:]

train_orig = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")
train_orig['target'] = np.where(train_orig['target'] >= 0.5, True, False)


#train_df, validate_df = train_test_split(train, test_size=0.1)
print('%d train comments, %d validate comments' % (len(train), len(test)))

MAX_FEATURES = 100000
EMBED_SIZE = 300
MAX_LEN = 220
TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'

tokenizer = Tokenizer(num_words=MAX_FEATURES, lower=True)
tokenizer.fit_on_texts(train[TEXT_COLUMN])
word_index = tokenizer.word_index


train_text = tokenizer.texts_to_sequences(train[TEXT_COLUMN])
train_text = pad_sequences(train_text, maxlen=MAX_LEN)
train_labels = train_orig[TOXICITY_COLUMN]
test_text = tokenizer.texts_to_sequences(test[TEXT_COLUMN])
test_text = pad_sequences(test_text, maxlen=MAX_LEN)

del([df, train, test, train_orig])
gc.collect()


EMBED_PATHS = [ '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
                '../input/glove840b300dtxt/glove.840B.300d.txt']

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)

def build_embedding_matrix(word_index, path):

    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, EMBED_SIZE))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
        except:
            embedding_matrix[i] = embedding_index["unknown"]
            
    del embedding_index
    gc.collect()
    return embedding_matrix

def build_embeddings(word_index):

    embedding_matrix = np.concatenate(
        [build_embedding_matrix(word_index, f) for f in EMBED_PATHS], axis=-1) 
    return embedding_matrix

embedding_matrix = build_embeddings(word_index)

# %% [code]
# add attention layer
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
    
    
# define the network
check_point = ModelCheckpoint("best_model.hdf5", monitor = "val_loss", mode = "min",
                              save_best_only = True, verbose = 1)
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 1)


def build_model(lr = 0.0, lr_d = 0.0, dr = 0.0):
    w = Input(shape = (MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights = [embedding_matrix], trainable = False)(w)
    x = SpatialDropout1D(dr)(x)
    
    att = Attention(MAX_LEN)(x)
    
    x = Bidirectional(CuDNNGRU(256, return_sequences = True))(x)
    x = Conv1D(128, kernel_size = 3, padding = "same", kernel_initializer = "he_uniform")(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    x = concatenate([att, avg_pool, max_pool])
    x = Dropout(dr)(x)
    x = Dense(1, activation = "sigmoid")(x)
    
    model = Model(inputs = w, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    
    return model


# run network
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=1337)
pred = np.zeros((len(test_text), 1))
n_fold = 0
for idx_train, idx_valid in rskf.split(train_text, train_labels):
    train_x, train_y = train_text[idx_train], train_labels[idx_train]
    valid_x, valid_y = train_text[idx_valid], train_labels[idx_valid]
    
    model = build_model(lr = 0.01, lr_d = 0.001, dr = 0.2)
    model.fit(train_x, train_y, validation_data=(valid_x, valid_y), 
              batch_size = 256, epochs = 5, verbose = 1, 
              callbacks = [check_point, early_stop])
    oof_preds = model.predict(valid_x)
    pred += model.predict(test_text, batch_size = 1024, verbose = 1)/4
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds)))
    n_fold = n_fold + 1
 
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
submission['prediction'] = pred
submission.to_csv('submission.csv', index=False)
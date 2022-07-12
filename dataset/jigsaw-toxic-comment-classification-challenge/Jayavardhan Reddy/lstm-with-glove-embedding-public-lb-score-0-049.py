'''
Single model may achieve LB scores at around 0.050
Don't need to be an expert of feature engineering
All you need is a GPU!!!!!!!

The code is tested on Keras 2.0.0 using Tensorflow backend, and Python 3.5

referrence Code:https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
'''

########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

import sys

########################################
## set directories and parameters
########################################
#INPUT_DATA_DIR='../input/'

#EMBEDDING_FILE=INPUT_DATA_DIR+'glove.6B.100d.txt'
#TRAIN_DATA_FILE=INPUT_DATA_DIR+'train.csv'
#TEST_DATA_FILE=INPUT_DATA_DIR+'test.csv'

path = '../input/'
comp = 'jigsaw-toxic-comment-classification-challenge/'
EMBEDDING_FILE=f'{path}glove6b50d/glove.6B.50d.txt'
TRAIN_DATA_FILE=f'{path}{comp}train.csv'
TEST_DATA_FILE=f'{path}{comp}test.csv'

MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1

num_lstm = 100
num_dense = 50
rate_drop_lstm = 0.25
rate_drop_dense = 0.25

act = 'relu'

########################################
## index word vectors
########################################
print('Indexing word vectors')

#Glove Vectors
embeddings_index = {}
f = open(EMBEDDING_FILE)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

########################################
## process texts in datasets
########################################
print('Processing text dataset')

#Regex to remove all Non-Alpha Numeric and space
special_character_removal=re.compile(r'[^a-z\d ]',re.IGNORECASE)

#regex to replace all numerics
replace_numbers=re.compile(r'\d+',re.IGNORECASE)

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)
    
    #Remove Special Characters
    text=special_character_removal.sub('',text)
    
    #Replace Numbers
    text=replace_numbers.sub('n',text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)


list_sentences_train = train_df["comment_text"].fillna("NA").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train_df[list_classes].values
list_sentences_test = test_df["comment_text"].fillna("NA").values


comments = []
for text in list_sentences_train:
    comments.append(text_to_wordlist(text))
    
test_comments=[]
for text in list_sentences_test:
    test_comments.append(text_to_wordlist(text))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(comments + test_comments)

sequences = tokenizer.texts_to_sequences(comments)
test_sequences = tokenizer.texts_to_sequences(test_comments)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', y.shape)

test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of test_data tensor:', test_data.shape)

########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


########################################
## sample train/validation data
########################################
#np.random.seed(1234)
perm = np.random.permutation(len(data))
idx_train = perm[:int(len(data)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data)*(1-VALIDATION_SPLIT)):]

data_train=data[idx_train]
labels_train=y[idx_train]
print(data_train.shape,labels_train.shape)

data_val=data[idx_val]
labels_val=y[idx_val]

print(data_val.shape,labels_val.shape)

########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences= embedding_layer(comment_input)
x = lstm_layer(embedded_sequences)
merged = Dropout(rate_drop_dense)(x)
merged = BatchNormalization()(merged)
merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)
preds = Dense(6, activation='sigmoid')(merged)

########################################
## train the model
########################################
model = Model(inputs=[comment_input], \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['accuracy'])
model.summary()

STAMP = 'simple_lstm_glove_vectors_%.2f_%.2f'%(rate_drop_lstm,rate_drop_dense)
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit(data_train, labels_train, \
        validation_data=(data_val, labels_val), \
        epochs=20, batch_size=200, shuffle=True, \
         callbacks=[early_stopping, model_checkpoint])
         
model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])
         
#######################################
## make the submission
########################################
print('Start making the submission before fine-tuning')

y_test = model.predict([test_data], batch_size=8192, verbose=1)

sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission[list_classes] = y_test

sample_submission.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)
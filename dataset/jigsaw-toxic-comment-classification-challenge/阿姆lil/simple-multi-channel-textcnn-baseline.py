import sys, os, re, csv, codecs, gc, numpy as np, \
pandas as pd, pickle as pkl, tensorflow as tf
#=================Keras==============
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Conv1D, Conv2D, \
Embedding, Dropout, Activation, Permute
from keras.layers import Bidirectional, MaxPooling1D, MaxPooling2D, \
Reshape, Flatten, concatenate, BatchNormalization, GlobalMaxPool1D, \
GlobalMaxPool2D
from keras import backend
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers, backend
#=================nltk===============
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
#=================gensim=============
import gensim
#=================save_list==========
# import pickle # to save data for time

path = '../input/'
comp = 'jigsaw-toxic-comment-classification-challenge/'
EMBEDDING_FILE=f'{path}glove6b50d/glove.6B.50d.txt'
TRAIN_DATA_FILE=f'{path}{comp}train.csv'
TEST_DATA_FILE=f'{path}{comp}test.csv'

embed_size = 50 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 110 # max number of words in a comment to use
number_filters = 100 # the number of CNN filters

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values

comments = list_sentences_train
test_comments = list_sentences_test
# tokenize
tokenizer = Tokenizer(num_words=max_features,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'', lower=True)

tokenizer.fit_on_texts(list(list(comments) + list(test_comments)))
comments_sequence = tokenizer.texts_to_sequences(comments)
test_comments_sequence = tokenizer.texts_to_sequences(test_comments)    
X_t = pad_sequences(comments_sequence , maxlen=maxlen)
X_te = pad_sequences(test_comments_sequence, maxlen=maxlen)

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
# filter_size
filter_size = [3, 4, 5]

inp = Input(shape=(maxlen, ))
x1 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
x2 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
x1 = Reshape((110, 50, 1))(x1)
x2 = Reshape((110, 50, 1))(x2)
x = concatenate([x1, x2])

# Version of Conv1D
# conv_blocks = []
# for sz in filter_size:
#     conv = Conv1D(number_filters, sz)(x)
#     batch_norm = BatchNormalization()(conv)
#     activation = Activation('elu')(batch_norm)
#     pooling = GlobalMaxPool1D()(activation)
#     conv_blocks.append(pooling)

# Version of Conv2D
conv_blocks = []
for sz in filter_size:
    conv = Conv2D(number_filters, (sz, embed_size), data_format='channels_last')(x)
    batch_norm = BatchNormalization()(conv)
    activation = Activation('elu')(batch_norm)
    pooling = GlobalMaxPool2D()(activation)
    conv_blocks.append(pooling)
    
x = concatenate(conv_blocks)
print(x.shape)
x = Dense(6, activation="sigmoid")(x)
print(x.shape)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_t, y, batch_size=2048, epochs=3)

y_test = model.predict([X_te], batch_size=512, verbose=1)
sample_submission = pd.read_csv(f'{path}{comp}sample_submission.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv('submission_textcnn.csv', index=False)
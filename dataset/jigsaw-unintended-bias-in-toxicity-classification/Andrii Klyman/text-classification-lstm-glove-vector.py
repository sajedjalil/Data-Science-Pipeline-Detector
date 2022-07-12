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
import pandas as pd
import time

start_time = time.time()

train  = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv').fillna(' ') #.sample(10000)
test   = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv').fillna(' ') #.sample(10000)
#sample = pd.read_csv('sample_submission.csv')

from tqdm import tqdm_notebook, tnrange
from tqdm.auto import tqdm
tqdm.pandas(desc='Progress')

train['total_length']   = train['comment_text'].progress_apply(len)
train['capitals']       = train['comment_text'].progress_apply(lambda comment: sum(1 for c in comment if c.isupper()))
train['caps_vs_length'] = train.progress_apply(lambda row: float(row['capitals'])/float(row['total_length']), axis=1)
train['num_words']      = train.comment_text.str.count('\S+')
train['spec_symbols']   = train['comment_text'].progress_apply(lambda comment: sum(1 for c in comment if c in ['!','?',":","-"]))
train['spec_symbols_per_length']   = train.progress_apply(lambda row: float(row['spec_symbols'])/float(row['total_length']), axis=1)


test['total_length']   = test['comment_text'].progress_apply(len)
test['capitals']       = test['comment_text'].progress_apply(lambda comment: sum(1 for c in comment if c.isupper()))
test['caps_vs_length'] = test.progress_apply(lambda row: float(row['capitals'])/float(row['total_length']), axis=1)
test['num_words']      = test.comment_text.str.count('\S+')
test['spec_symbols']   = test['comment_text'].progress_apply(lambda comment: sum(1 for c in comment if c in ['!','?',":","-"]))
test['spec_symbols_per_length']   = test.progress_apply(lambda row: float(row['spec_symbols'])/float(row['total_length']), axis=1)

# ------------------------------------------------------

from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.stem.lancaster import LancasterStemmer
lc = LancasterStemmer()
from nltk.corpus import stopwords 
from nltk.stem import SnowballStemmer
sb = SnowballStemmer("english")

import gc
gc.enable()

train_text = train['comment_text']
test_text  = test['comment_text']
text_list  = pd.concat([train_text, test_text])
y          = train['target'].values

num_train_data = y.shape[0]
stop_words = set(stopwords.words('english')) 

word_dictionary = {}
word_index      = 1
word_sequences  = []
for doc in tqdm(text_list):
  word_seq = []
  for token in doc.split():
    if (token in stop_words): continue
    if (token not in word_dictionary): 
      word_dictionary[token] = word_index
      word_index +=1
    word_seq.append(word_dictionary[token])
  word_sequences.append(word_seq)
# -----------------------------------    
# word_dictionary: слово - индекс
# word_sequences: очередность индексов вместо слов
del text_list
gc.collect()
# -----------------------------------   

def load_glove(word_dict):
  EMBEDDING_FILE = '../input/glove-840b-300d/glove.840B.300d.txt'
  def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
  embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)
  embed_size = 300
  nb_words = len(word_dict)+1
  embedding_matrix = np.zeros((nb_words, embed_size), dtype=np.float32)
  unknown_vector = np.zeros((embed_size,), dtype=np.float32) - 1.
  print(unknown_vector[:5])
  for key in tqdm(word_dict):
        word = key
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.upper()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = ps.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lc.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = sb.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        # тут можно попробовать по букве с конца обрезать или корень слова взять
        embedding_matrix[word_dict[key]] = unknown_vector
        
  return embedding_matrix, nb_words        
# ----------------------------------------- 

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.layers import Dense, CuDNNLSTM, Bidirectional
from keras.layers import SpatialDropout1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import backend as K
# -----------------------------------------
def build_nn(word_dictionary, embed_size, max_length): # , embedding_matrix
    inp   = Input(shape=(max_length,))
    #x     = Embedding(nb_words,  embed_size,  trainable=False)(inp) # weights=[embedding_matrix],
    #drop  = SpatialDropout1D(0.3)(x)
    lstm  = Bidirectional(CuDNNLSTM(256, return_sequences=True))(inp)
    #gru   = Bidirectional(CuDNNGRU(128, return_sequences=True))(lstm)    
    #pool1 = GlobalMaxPooling1D()(lstm)
    pool2 = GlobalMaxPooling1D()(lstm)
    #conc  = Concatenate()([pool1, pool2])
    predi = Dense(1, activation='sigmoid')(pool2)
    model = Model(inputs=inp, outputs=predi)
    adam  = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])    
    print(model.summary())
    return model

# -----------------------------------------

train_word_sequences = word_sequences[:num_train_data]
test_word_sequences = word_sequences[num_train_data:]

print("--- %s seconds ---" % int(time.time() - start_time))

# embedding_matrix_glove, nb_words = load_glove(word_dictionary)

del word_sequences
gc.collect()

max_length = 55
embed_size = 300
learning_rate = 0.001

train_word_sequences = pad_sequences(train_word_sequences, maxlen=max_length, padding='post')
test_word_sequences = pad_sequences(test_word_sequences, maxlen=max_length, padding='post')
# print(train_word_sequences)  [1,2,37,64 ...0, 0, 0]          

# -----------------------------------------

model = build_nn(word_dictionary, embed_size, max_length) # , embedding_matrix_glove
model.fit(train_word_sequences, y, batch_size=512, epochs=2)
y_valid_pred = np.squeeze(model.predict(test_word_sequences, batch_size=512, verbose=2))

# -----------------------------------------

df_submit = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv') #.sample(10000)
df_submit.prediction = y_valid_pred
df_submit.to_csv('submission.csv', index=False)

# -----------------------------------------
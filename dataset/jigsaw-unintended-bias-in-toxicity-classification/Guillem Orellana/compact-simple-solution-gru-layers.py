# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import tensorflow as tf

np.random.seed(42)

EMBEDDING_DIMS = 300
MAX_SEQ_LEN = 200

df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
for col in identity_columns:
    del df[col]

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(list(df.comment_text) + list(test_df.comment_text))

sequences = tokenizer.texts_to_sequences(df.comment_text)

padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(maxlen=MAX_SEQ_LEN, sequences=sequences, padding="post", truncating='post')

embeddings_index = {}
with open('../input/glove840b300dtxt/glove.840B.300d.txt') as f:
    for line in f:
        values = line.split(' ')
        word = values[0] ## The first entry is the word
        coefs = np.asarray(values[1:], dtype='float32') ## These are the vecotrs representing the embedding for the word
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDING_DIMS))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, EMBEDDING_DIMS, input_length=MAX_SEQ_LEN, weights=[embedding_matrix], trainable=False),
    tf.keras.layers.CuDNNGRU(128, return_sequences=True),
    tf.keras.layers.CuDNNGRU(64, return_sequences=True),
    tf.keras.layers.CuDNNGRU(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(padded_sequences, np.where(df.target >= .5, True, False), batch_size=128, epochs=3)


test_sequences = tokenizer.texts_to_sequences(test_df.comment_text.values)
pad_test_sequences = tf.keras.preprocessing.sequence.pad_sequences(maxlen=MAX_SEQ_LEN, sequences=test_sequences, padding="post", truncating='post')
predictions = model.predict(pad_test_sequences)

submissions_df = pd.DataFrame({'id': test_df.id, 'prediction': predictions.flatten() })
submissions_df.to_csv('submission.csv', index=False)

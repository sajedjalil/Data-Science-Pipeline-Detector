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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM

from sklearn.model_selection import train_test_split
from sklearn import metrics

max_len = 100
max_features = 50000
embed_size = 300


def load_and_prec(val_size = 0.1):
    df_train = pd.read_csv("../input/train.csv")
    df_train, df_val = train_test_split(df_train, test_size=val_size, random_state=666)
    df_test = pd.read_csv("../input/test.csv")

    # 填上缺失值
    x_train = df_train["question_text"].fillna("_##_").values
    x_val = df_val["question_text"].fillna("_##_").values
    x_test = df_test["question_text"].fillna("_##_").values

    # 转成token
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(x_train))
    x_train = tokenizer.texts_to_sequences(x_train)
    x_val = tokenizer.texts_to_sequences(x_val)
    x_test = tokenizer.texts_to_sequences(x_test)

    # 补0
    x_train = pad_sequences(x_train, maxlen=max_len)
    x_val = pad_sequences(x_val, maxlen=max_len)
    x_test = pad_sequences(x_test, maxlen=max_len)

    # target
    y_train = df_train["target"].values
    y_val = df_val["target"].values

    return x_train, y_train, x_val, y_val, x_test, tokenizer.word_index


def load_glove(word_index, max_features):
    EMBEDDING_FILE = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embedding_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, "r", encoding="utf-8"))

    all_embs = np.stack(embedding_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))

    for word ,i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def loadLstmodel(max_features, embed_size, embedding_matrix=None):
    model = Sequential()
    if embedding_matrix is None:
        model.add(Embedding(max_features, embed_size))
    else:
        model.add(Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False))
    model.add(LSTM(64, activation='tanh', return_sequences=False))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    return model

def find_best_prob(model, x_val, y_val, batch_size=512):
    y_pred = model.predict([x_val], batch_size=batch_size, verbose=1)
    best_f = 0
    best_p = 0
    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        f1 = metrics.f1_score(y_val, (y_pred > thresh).astype(int))
        if f1 > best_f:
            best_f = f1
            best_p = thresh
        print("f1 score at threshold{0} is {1}".format(thresh, f1))
    print("best param:", best_p, "f1:", best_f)
    return best_p


if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, word_index = load_and_prec()
    embedding_matrix = load_glove(word_index, max_features)
    model = loadLstmodel(max_features, embed_size, embedding_matrix=embedding_matrix)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=512, epochs=2, verbose=1)
    
    best_prob = find_best_prob(model, x_val, y_val, batch_size=512)
    y_pred = model.predict(x_test)
    
    sub = pd.read_csv("../input/sample_submission.csv")
    sub.prediction = y_pred > best_prob
    sub.to_csv("submission_csv", index=False)

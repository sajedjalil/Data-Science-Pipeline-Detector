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
import numpy as np
import re
from tqdm import tqdm

from nltk.corpus import stopwords

from gensim.models import Word2Vec
from gensim.models import Phrases
from gensim.models.phrases import Phraser

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Dropout,Bidirectional, Reshape, Flatten, CuDNNGRU, CuDNNLSTM
from keras.models import Model, Sequential
from keras.initializers import Constant
from keras import backend as K

#f1 스코어
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

# 불용어 만들기
stopWords = stopwords.words('english')

# 데이터 정제
def cleanData(sentence):
    processedList = ""
    
    # convert to lowercase, ignore all special characters - keep only alpha-numericals and spaces (not removing full-stop here)
    sentence = re.sub(r'[^A-Za-z0-9\s.]',r'',str(sentence).lower())
    sentence = re.sub(r'\n',r' ',sentence)
    
    # remove stop words
    sentence = " ".join([word for word in sentence.split() if word not in stopWords])
    
    return sentence


train_df['question_text'] = train_df['question_text'].apply(lambda x :cleanData(x))
test_df['question_text'] = test_df['question_text'].apply(lambda x :cleanData(x))
train_X = train_df['question_text']
test_X = test_df['question_text']
print(train_X.shape)
print(test_X.shape)

# corpus 만들기
tmp_corpus = train_X.apply(lambda x: x.split("."))

corpus = []
for i in range(len(tmp_corpus)):
    for line in tmp_corpus[i]:
        words = [x for x in line.split()]
        corpus.append(words)

#keras로 전처리
maxlen = 70 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)
test_X = tokenizer.texts_to_sequences(test_X)

train_X = pad_sequences(train_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

train_y = train_df['target'].values
train_y = np_utils.to_categorical(train_y)

#gensim으로 Word2Vec 만들기
model = Word2Vec(corpus, sg =1, window = 3, size = 100, min_count = 5, workers = 4 , iter = 50)
filename = '../input/gensim_word2vec.txt'
model.wv.save_word2vec_format(filename, binary = False)

#Word2Vec 사용하기
import os
embedding_index = {}
f = open(os.path.join("",filename), encoding = 'utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embedding_index[word] = coefs
f.close()

word_index = tokenizer.word_index

num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, 100))

for word, i in word_index.items():
    if i > num_words:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
#model 만들기
model1 = Sequential()
model1.add(Embedding(num_words, 100, embeddings_initializer= Constant(embedding_matrix),trainable=False, input_length = 70))
model1.add(Bidirectional(CuDNNLSTM(150, return_sequences=True)))
model1.add(Bidirectional(CuDNNLSTM(125, return_sequences=True)))
model1.add(Bidirectional(CuDNNLSTM(100, return_sequences=True)))
model1.add(Flatten())
model1.add(Dense(100, activation = 'relu'))
model1.add(Dense(50, activation = 'relu'))
model1.add(Dense(2, activation = 'sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
#model1.summary()
model1.fit(train_X, train_y, batch_size = 500, epochs = 7)


pred_y = np.argmax(model1.predict(test_X), axis = 1)
test_df = pd.read_csv("../input/test.csv", usecols=["qid"])
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_y
out_df.to_csv("submission.csv", index=False)
import os
import time
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNLSTM, Conv1D,Dropout
from keras.layers import Bidirectional, GlobalMaxPool1D,MaxPooling1D,BatchNormalization,Flatten,RepeatVector,Permute,concatenate
from keras.models import Model

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train, dev = train_test_split(train, test_size=0.1, random_state=2018)

embed_size = 300 
max_features = 50000 
maxlen = 40 

train_X = train["question_text"].fillna("_na_").values
dev_X = dev["question_text"].fillna("_na_").values
test_X = test["question_text"].fillna("_na_").values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
dev_X = tokenizer.texts_to_sequences(dev_X)
test_X = tokenizer.texts_to_sequences(test_X)

train_X = pad_sequences(train_X, maxlen=maxlen)
dev_X = pad_sequences(dev_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

train_y = train['target'].values
dev_y = dev['target'].values

EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
        
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def net(input_shape):
    sentence_indices = Input(input_shape, dtype='int32')
    X = Embedding(max_features, embed_size,weights=[embedding_matrix])(sentence_indices)
    activations = Bidirectional(CuDNNLSTM(128, return_sequences=True))(X)
    activations = Dropout(0.1)(activations)
    # compute importance for each step
    attention = Dense(1, activation='tanh')(activations)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(128)(attention)
    attention = Permute([2, 1])(attention)

    X = concatenate([activations, attention])
    activations = Bidirectional(CuDNNLSTM(128, return_sequences=True))(X)
    activations = Dropout(0.1)(activations)
    # compute importance for each step
    attention = Dense(1, activation='tanh')(activations)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(128)(attention)
    attention = Permute([2, 1])(attention)

    X = concatenate([activations, attention])
    X = Bidirectional(CuDNNLSTM(64, return_sequences=True))(X)
    X = Dropout(0.1)(X)
    X = GlobalMaxPool1D()(X)
    X = Dense(512, activation="relu")(X)
    X = Dropout(0.1)(X)
    X = Dense(512, activation="relu")(X)
    X = Dropout(0.1)(X)
    X = Dense(512, activation="relu")(X)
    X = Dropout(0.1)(X)
    X = Dense(1, activation="sigmoid")(X)
    model = Model(inputs=sentence_indices, outputs=X)
    return model
model=net((maxlen,))
#model.layers[1].trainable=False
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1])
print(model.summary())
weight={0:0.4,1:1}
model.fit(train_X, train_y, batch_size=512, epochs=2 , validation_data=(dev_X, dev_y),class_weight=weight)
model.fit(train_X, train_y, batch_size=512, epochs=1 , validation_data=(dev_X, dev_y),class_weight=weight)


f1=[]
pred_noemb_val_y = model.predict([dev_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.4, 0.702, 0.01):
    thresh = np.round(thresh, 2)
    f1.append(metrics.f1_score(dev_y, (pred_noemb_val_y>thresh).astype(int)))
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(dev_y, (pred_noemb_val_y>thresh).astype(int))))
f1=np.array(f1)

thresh=np.argmax(f1)*0.01+0.4
print(thresh)
pred_noemb_val_y1 = model.predict([test_X], verbose=1)
label1=(pred_noemb_val_y1>thresh).astype(int)
model=None
test_labels=label1
submission=pd.read_csv("../input/sample_submission.csv")
submission["prediction"]=test_labels
submission.head()
submission.to_csv("submission.csv",index=False)
print("ok")



# Fork of Sergei Fironov's script CNN GLOVE300 3-OOF 3 epochs

import os
os.environ['OMP_NUM_THREADS'] = '4'

import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Embedding, Input, Concatenate, Conv1D, Activation, TimeDistributed, Flatten, RepeatVector, Permute,multiply
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, GRU, GlobalAveragePooling1D, MaxPooling1D, SpatialDropout1D, BatchNormalization
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re 
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

print('loading embeddings vectors')
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(' ')) for o in open('../input/glove840b300dtxt/glove.840B.300d.txt'))

min_count = 10 #the minimum required word frequency in the text
max_features = 27403 #it's from previous run with min_count=10
maxlen = 150 #padding length
num_folds = 5 #number of folds
batch_size = 512 
epochs = 4
embed_size = 300 #embeddings dimension

sia = SentimentIntensityAnalyzer()

train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
test = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv")

list_sentences_train = train["comment_text"].fillna("").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("").values

print('mean text len:',train["comment_text"].str.count('\S+').mean())
print('max text len:',train["comment_text"].str.count('\S+').max())

#tokenizer = Tokenizer()
#tokenizer.fit_on_texts(list(list_sentences_train)) #  + list(list_sentences_test)
#num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])
#print('num_words',num_words)
#max_features = num_words
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train)) # + list(list_sentences_test)
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
print('padding sequences')
X_train = {}
X_test = {}
X_train['text'] = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen, padding='post', truncating='post')
X_test['text'] = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen, padding='post', truncating='post')

print('numerical variables')
train['num_words'] = train.comment_text.str.count('\S+')
test['num_words'] = test.comment_text.str.count('\S+')
train['num_comas'] = train.comment_text.str.count('\.')
test['num_comas'] = test.comment_text.str.count('\.')
train['num_bangs'] = train.comment_text.str.count('\!')
test['num_bangs'] = test.comment_text.str.count('\!')
train['num_quotas'] = train.comment_text.str.count('\"')
test['num_quotas'] = test.comment_text.str.count('\"')
train['avg_word'] = train.comment_text.str.len() / (1 + train.num_words)
test['avg_word'] = test.comment_text.str.len() / (1 + test.num_words)
#print('sentiment')
#train['sentiment'] = train.comment_text.apply(lambda s : sia.polarity_scores(s)['compound'])
#test['sentiment'] = test.comment_text.apply(lambda s : sia.polarity_scores(s)['compound'])
scaler = MinMaxScaler()
X_train['num_vars'] = scaler.fit_transform(train[['num_words','num_comas','num_bangs','num_quotas','avg_word']])
X_test['num_vars'] = scaler.transform(test[['num_words','num_comas','num_bangs','num_quotas','avg_word']])

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()

print('create embedding matrix')
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

def get_model_cnn(X_train):
    global embed_size
    inp = Input(shape=(maxlen, ), name="text")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    z = GlobalMaxPool1D()(x)
    x = GlobalMaxPool1D()(Conv1D(embed_size, 4, activation="relu")(x))
    x = Concatenate()([x,z,num_vars])
    x = Dropout(0.3)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=[inp,num_vars], outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model        

print('start modeling')
scores = []
predict = np.zeros((test.shape[0],6))
oof_predict = np.zeros((train.shape[0],6))

kf = KFold(n_splits=num_folds, shuffle=True, random_state=239)
for train_index, test_index in kf.split(X_train['num_vars']):
    kfold_X_train = {}
    kfold_X_valid = {}
    y_train,y_test = y[train_index], y[test_index]
    for c in ['text','num_vars']:
        kfold_X_train[c] = X_train[c][train_index]
        kfold_X_valid[c] = X_train[c][test_index]

    model = get_model_cnn(X_train)
    model.fit(kfold_X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    predict += model.predict(X_test, batch_size=1000) / num_folds
    oof_predict[test_index] = model.predict(kfold_X_valid, batch_size=1000)
    cv_score = roc_auc_score(y_test, oof_predict[test_index])
    scores.append(cv_score)
    print('score: ',cv_score)

print('Total CV score is {}'.format(np.mean(scores)))    


sample_submission = pd.DataFrame.from_dict({'id': test['id']})
oof = pd.DataFrame.from_dict({'id': train['id']})
for c in list_classes:
    oof[c] = np.zeros(len(train))
    sample_submission[c] = np.zeros(len(test))
    
sample_submission[list_classes] = predict
sample_submission.to_csv('submit_cnn_avg_' + str(num_folds) + '_folds.csv', index=False)

oof[list_classes] = oof_predict
oof.to_csv('cnn_'+str(num_folds)+'_oof.csv', index=False)
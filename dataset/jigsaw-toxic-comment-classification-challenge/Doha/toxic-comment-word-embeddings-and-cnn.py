# Toxic comment model [word embedding + CNN with multiple kernels]
import pandas as pd
import numpy as np
import keras
from keras.layers import concatenate,Input,Dense, GlobalAveragePooling1D, Embedding, Dropout
from keras.layers import LSTM, Flatten, Activation, Conv1D,SpatialDropout1D,GRU,GlobalMaxPooling1D
from keras.models import Sequential,Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from keras.callbacks import Callback
import re
import nltk
from nltk.corpus import wordnet

######## Read fast text data
WORDS = {}
index = [1]

def get_coefs(word, *arr): 
    WORDS[word] = index[0]
    index[0] = index[0]+1
    return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open('../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'))

######## Spelling correction
def P(word): 
    return - WORDS.get(word, 0)

def correction(word): 
    return max(candidates(word), key=P)

def candidates(word): 
    return (known([word]) or known(edits1(word)) or [word])

def known(words): 
    return set(w for w in words if w in WORDS)

def edits1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

######## Read challenge data
train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv') 
print(train.head())
cols  = list(train.columns.values)
cols.remove('id')
cols.remove('comment_text')
print(cols)

######## Correct misspelling in both train and test data
def corrected_words(row):
    comment = row['comment_text']
    result =  ' '
    for w in nltk.word_tokenize(comment):
        if len(w)>1 and not wordnet.synsets(w):
            result += correction(w) + ' '
        else:
            result += w + ' '
    
    return result

train['comment_text'] = train.apply(corrected_words, axis=1)
test['comment_text'] = test.apply(corrected_words, axis=1)


######## text tokenization
max_features = 25000
max_len=100

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train['comment_text'])
tokenizer.fit_on_texts(test['comment_text'])
def process_text(data):
    sequences = tokenizer.texts_to_sequences(data)
    return pad_sequences(sequences=sequences, maxlen=max_len)
 
train_data = process_text(train['comment_text'])
test_data = process_text(test['comment_text'])
input_dim = np.max([np.max(train_data), np.max(test_data)]) + 1
embedding_dims = 300
print(input_dim)

######## Load pretrained embedding data
word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embedding_dims))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

######## Build Model [Embedding and CNN]
def clf_model():
    seq = Input(shape=(max_len,), name='seq')
    l = Embedding(max_features, embedding_dims, weights=[embedding_matrix])(seq)
    l = SpatialDropout1D(0.4)(l)
    c2 = Conv1D(embedding_dims,2, activation='relu')(l)
    c3 = Conv1D(embedding_dims,3, activation='relu')(l)
    c4 = Conv1D(embedding_dims,4, activation='relu')(l)
    c5 = Conv1D(embedding_dims,5, activation='relu')(l)
    max_pool2 = GlobalMaxPooling1D()(c2)
    max_pool3 = GlobalMaxPooling1D()(c3)
    max_pool4 = GlobalMaxPooling1D()(c4)
    max_pool5 = GlobalMaxPooling1D()(c5)
    conc = concatenate([max_pool2,max_pool3,max_pool4,max_pool5])
    conc = Dense(36)(conc)
    output = Dense(6, activation='sigmoid', name='output')(conc)
    model = Model(inputs=[seq], outputs=output)
    model.compile(loss='binary_crossentropy',optimizer='adam')
    return model

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
            
print("starting")
x_train, x_test, y_train, y_test = train_test_split(train_data, train[cols], test_size = 0.05, random_state = 144)
RocAuc = RocAucEvaluation(validation_data=(x_test, y_test), interval=1)
text_clf = clf_model()
submission = pd.DataFrame()
submission['id'] = test.id
text_clf.fit(x_train, y_train, batch_size=50, epochs=2, validation_data=(x_test, y_test), callbacks=[RocAuc], verbose=0)
pred = text_clf.predict(test_data)
i=0
for col in cols:
    submission[col] = pred[:,i]
    i+=1
submission.to_csv('submission_cnn.csv', index=False)
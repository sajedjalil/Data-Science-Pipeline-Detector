import numpy as np
import pandas as pd
import re
import io
import time
import gc
import sys
import codecs
import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.cluster import MiniBatchKMeans
from textblob import TextBlob
from nltk.stem.porter import PorterStemmer
from collections import Counter
from fastcache import clru_cache as lru_cache
from scipy.sparse import lil_matrix, hstack, vstack
from multiprocessing import Pool
import tensorflow as tf
import keras as ks
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

random.seed(0)
np.random.seed(0)

train_features = 120000
emdded_size = 512
train_maxlen = 50
train_minlen = 15
nth_fat_data = 12
training_epochs = 5
n_estimaters = 3
n_word_bins = 22000
use_split_question = False

debug = False
validation = False

print( 'read data' )
train_df = pd.read_csv( '../input/train.csv' )
test_df = pd.read_csv( '../input/test.csv' )
if debug:
    train_df = train_df.iloc[:120000]
    test_df = test_df.iloc[:10000]

df = pd.concat( [ train_df, test_df ], axis=0, sort=False, ignore_index=True )
y_train = train_df[ 'target' ].values.astype(np.int32)
ids_test = test_df[ 'qid' ].values
train_size = len(train_df)
del train_df, test_df

print( 'make feature' )

idx_fold = np.array_split(np.arange(len(df)),4)

def make_feature_one(texts):
    result = np.zeros( (len(texts),28), dtype=np.float32 )
    for i,text in enumerate(texts):
        result[i,0] = len(text)
        result[i,1] = text.count(' ')
        result[i,2] = text.count('.')
        result[i,3] = text.count('!')
        result[i,4] = text.count('?')
        result[i,5] = text.count(',')
        result[i,6] = max(text.count(' '), 1)
        result[i,7] = len(set(w for w in text.split()))
        result[i,8] = sum([1 for c in text if c.isupper()])
        result[i,9] = sum([1 for c in text if c.isnumeric()])
        result[i,10] = result[i,1] / result[i,0]
        result[i,11] = result[i,2] / result[i,0]
        result[i,12] = result[i,3] / result[i,0]
        result[i,13] = result[i,4] / result[i,0]
        result[i,14] = result[i,5] / result[i,0]
        result[i,15] = result[i,6] / result[i,0]
        result[i,16] = result[i,7] / result[i,0]
        result[i,17] = result[i,8] / result[i,0]
        result[i,18] = result[i,9] / result[i,0]
        result[i,19] = result[i,1] / result[i,6]
        result[i,20] = result[i,2] / result[i,6]
        result[i,21] = result[i,3] / result[i,6]
        result[i,22] = result[i,4] / result[i,6]
        result[i,23] = result[i,5] / result[i,6]
        result[i,24] = result[i,6] / result[i,6]
        result[i,25] = result[i,7] / result[i,6]
        result[i,26] = result[i,8] / result[i,6]
        result[i,27] = result[i,9] / result[i,6]
    return result
    
def make_feature( n_fold ):
    t_df = df.iloc[idx_fold[n_fold]]
    texts = t_df[ 'question_text' ].values.tolist()
    return n_fold, make_feature_one(texts)

with Pool(4) as p:
    result = p.map(make_feature, np.arange(4))
stack_fold = [None,None,None,None]
for n_fold, mat in result:
    stack_fold[n_fold] = mat
feature = np.vstack(stack_fold)
feature_train = feature[:train_size]
feature_test = feature[train_size:]
del feature, stack_fold, result, n_fold, mat

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, ' %s '%punct)
    return x

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x

mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

print( 'make typical' )

mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

def make_typical( n_fold ):
    t_df = df.iloc[idx_fold[n_fold]]
    q = t_df["question_text"].apply(lambda x: replace_typical_misspell(clean_numbers(clean_text(x.lower()))))
    return n_fold, q.fillna("_##_").values.tolist()

with Pool(4) as p:
    result = p.map(make_typical, np.arange(4))
stack_fold = [None,None,None,None]
for n_fold, mat in result:
    stack_fold[n_fold] = mat
sequence = stack_fold[0]+stack_fold[1]+stack_fold[2]+stack_fold[3]
del stack_fold, result, n_fold, mat, df

def get_contvect():
    t1 = time.time()
    cnt = CountVectorizer()
    matrix_train  = cnt.fit_transform( sequence[:train_size] )
    matrix_test  = cnt.transform( sequence[train_size:] )
    print("end get_contvect %ssec"%(time.time()-t1))
    return matrix_train, matrix_test, cnt

def get_tfidfvect():
    t1 = time.time()
    cnt = TfidfVectorizer( max_features=n_word_bins )
    matrix_train  = cnt.fit_transform( sequence[:train_size] )
    matrix_test  = cnt.transform( sequence[train_size:] )
    print("end get_tfidfvect %ssec"%(time.time()-t1))
    return matrix_train, matrix_test, cnt

def get_tokenizer():
    t1 = time.time()
    tokenizer = Tokenizer(num_words=train_features)
    tokenizer.fit_on_texts(sequence[:train_size])
    print("end get_tokenizer %ssec"%(time.time()-t1))
    return tokenizer

def get_sentences():
    """ Note:
      If it is SINCERE question,
      thre is no insincere sentence in it.
      So long question can be slit.
      Both divided questions are sicere.
    """
    t1 = time.time()
    ptn = re.compile(r'[\.\?\!]')
    idx = np.arange(len(y_train))[y_train==0]
    add_sentence = []
    for i in idx:
        txt = sequence[i]
        sent = ptn.split(txt)
        if len(sent) > 1:
            sent = list(map(lambda x:x.strip(), sent))
            ml = min(list( map(lambda x:len(x), sent)))
            if ml > train_minlen:
                add_sentence.extend(sent)
    print("end get_sentences %ssec"%(time.time()-t1))
    return add_sentence

def make_parallel( n_fold ):
    if n_fold==0:
        return n_fold, get_tfidfvect()
    elif n_fold==1:
        return n_fold, get_contvect()
    elif n_fold==2:
        return n_fold, get_tokenizer()
    elif n_fold==3:
        return n_fold, get_sentences()

print( 'make parallel' )
ext_sentences = None
with Pool(4) as p:
    result = p.map(make_parallel, np.arange(4 if use_split_question else 3))
for n_fold, mat in result:
    if n_fold == 0:
        idfmatrix_train, idfmatrix_test, idfmatrix_cnt = mat
    elif n_fold == 1:
        cntmatrix_train, cntmatrix_test, cntmatrix_cnt = mat
    elif n_fold == 2:
        tokenizer = mat
    elif n_fold == 3:
        ext_sentences = mat

matrix_train = hstack([cntmatrix_train,idfmatrix_train]).tocsr()
matrix_test = hstack([cntmatrix_test,idfmatrix_test]).tocsr()
del cntmatrix_train, cntmatrix_test, idfmatrix_train, idfmatrix_test

word_index = tokenizer.word_index

def read_vector(fn):
    maxwc = len(word_index)
    wc = 0
    result = np.zeros((max(word_index.values())+1,300))
    t1 = time.time()
    with codecs.open(fn,'r','ascii','ignore') as f:
        ln = f.readline()
        while ln:
            t = ln.split()
            if len(t) >= 301:
                if t[-301] in word_index:
                    result[word_index[t[-301]]] = np.array(list(map(float,t[-300:])))
                    wc += 1
                    if wc >= maxwc:
                        break
            ln = f.readline()
    print("end read_vector %ssec"%(time.time()-t1))
    return result

def make_extra():
    t1 = time.time()
    feature_ext = make_feature_one(ext_sentences)
    feature_train_ext = np.vstack([feature_train,feature_ext])
    
    cntmatrix_ext = cntmatrix_cnt.transform(ext_sentences)
    idfmatrix_ext = idfmatrix_cnt.transform(ext_sentences)
    matrix_ext = hstack([cntmatrix_ext,idfmatrix_ext]).tocsr()
    matrix_train_ext = vstack([matrix_train,matrix_ext]).tocsr()
    
    y_train_ext = np.hstack([y_train,np.zeros(len(ext_sentences),)])
    
    x_train = sequence[:train_size] + ext_sentences
    x_test = sequence[train_size:]
    seq_train = tokenizer.texts_to_sequences(x_train)
    seq_train = pad_sequences(seq_train, maxlen=train_maxlen)
    seq_test = tokenizer.texts_to_sequences(x_test)
    seq_test = pad_sequences(seq_test, maxlen=train_maxlen)
    print("end make_extra %ssec"%(time.time()-t1))
    return seq_train, seq_test, matrix_train_ext, feature_train_ext, y_train_ext

def make_sequence( n_fold ):
    if n_fold == 0:
        if use_split_question:
            return n_fold, make_extra()
        else:
            x_train = sequence[:train_size]
            x_test = sequence[train_size:]
            seq_train = tokenizer.texts_to_sequences(x_train)
            seq_train = pad_sequences(seq_train, maxlen=train_maxlen)
            seq_test = tokenizer.texts_to_sequences(x_test)
            seq_test = pad_sequences(seq_test, maxlen=train_maxlen)
            return n_fold, (seq_train, seq_test, matrix_train, feature_train, y_train)
    elif n_fold == 1:
        return n_fold, read_vector("../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec")
    elif n_fold == 2:
        return n_fold, read_vector("../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt")
    elif n_fold == 3:
        return n_fold, read_vector("../input/embeddings/glove.840B.300d/glove.840B.300d.txt")
        
print( 'make sequence' )

with Pool(4) as p:
    result = p.map(make_sequence, np.arange(4))
word_vector = np.zeros((max(word_index.values())+1,600))
for n_fold, mat in result:
    if n_fold == 0:
        seq_train, seq_test, matrix_train, feature_train, y_train = mat
    elif n_fold == 1:
        word_vector[:,:300] += mat / 2.0
    elif n_fold == 2:
        word_vector[:,:300] += mat / 2.0
    elif n_fold == 3:
        word_vector[:,300:] += mat

del sequence, ext_sentences, cntmatrix_cnt, idfmatrix_cnt, tokenizer, result, n_fold, mat, idx_fold, train_size

class CyclicLR(Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
    
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

def fit_predict(train_X, ftrain_X, wtrain_X, train_y, val_X, fval_X, wval_X, val_y, callback=None):
    config = tf.ConfigProto(intra_op_parallelism_threads=4, use_per_session_threads=4, inter_op_parallelism_threads=4)
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        ks.backend.set_session(sess)
        inp = ks.Input(shape=(train_maxlen,))
        feat = ks.Input(shape=(ftrain_X.shape[1],))
        word = ks.Input(shape=(wtrain_X.shape[1],), sparse=True)

        x = ks.layers.Embedding(word_vector.shape[0], word_vector.shape[1], weights=[word_vector], trainable=False)(inp)
        x = ks.layers.SpatialDropout1D(0.1)(x)

        x = ks.layers.Bidirectional(ks.layers.CuDNNLSTM(60, return_sequences=True, return_state=True))(x)
        y = ks.layers.Bidirectional(ks.layers.CuDNNGRU(60, return_sequences=True, return_state=True))(x[0])

        avg_pool = ks.layers.GlobalAveragePooling1D()(y[0])
        max_pool = ks.layers.GlobalMaxPooling1D()(y[0])
        
        att = ks.layers.concatenate([x[1],x[2]])
        att = ks.layers.Dense(16, activation='sigmoid')(att)
        att = ks.layers.Dropout(0.5)(att)
        att = ks.layers.Dense(1, activation='sigmoid')(att)

        out = ks.layers.Dense(64, activation='sigmoid')(word)
        out = ks.layers.Dropout(0.5)(out)
        out = ks.layers.Dense(1, activation='sigmoid')(out)

        ft = ks.layers.Dense(2, activation='sigmoid')(feat)
        
        conc = ks.layers.concatenate([y[1], y[2], avg_pool, max_pool, feat])
        conc = ks.layers.Dense(12, activation='sigmoid')(conc)
        conc = ks.layers.Dropout(0.1)(conc)
        conc = ks.layers.concatenate([att, conc, ft, out])
        conc = ks.layers.Dense(16, activation="relu")(conc)
        conc = ks.layers.Dropout(0.1)(conc)
        outp = ks.layers.Dense(1, activation="sigmoid")(conc)    

        model = ks.Model(inputs=[inp, feat, word], outputs=outp)
        if val_y is None:
            model.compile(loss='binary_crossentropy', optimizer='adam')
            model.fit(x=[train_X, ftrain_X, wtrain_X], y=train_y, batch_size=512, epochs=training_epochs, callbacks = callback, verbose=0)
        else:
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
            model.fit(x=[train_X, ftrain_X, wtrain_X], y=train_y, batch_size=512, epochs=training_epochs, validation_data=([val_X,fval_X,wval_X], val_y), callbacks = callback, verbose=1)
        pred_test_y = model.predict([val_X, fval_X, wval_X], batch_size=10000, verbose=0)
        return pred_test_y

clr = CyclicLR(base_lr=0.001, max_lr=0.003,
               step_size=300., mode='exp_range',
               gamma=0.99994)

# Bagging emsemble
def run_one( train_X, ftrain_X, wtrain_X, train_Y, test_X, ftest_X, wtest_X, test_Y=None ):
    """ Note:
       In the bagging algorithm,
       in order to prevent almost the same model from being learned,
       randomly select words to use
    """
    np.random.seed(0)
    word_index = np.random.choice(np.arange(wtrain_X.shape[1]), n_word_bins, replace=False)
    word = wtrain_X[:,word_index]
    word_t = wtest_X[:,word_index]
    bag_index = np.random.permutation(len(train_X))
    #bag_index = np.random.choice(bag_index, len(train_X), replace=True)
    return fit_predict(train_X[bag_index], ftrain_X[bag_index], word[bag_index], train_Y[bag_index],
            test_X, ftest_X, word_t, test_Y, callback = [clr,])

def bestThresshold(y_train,train_preds):
    tmp = [0,0,0] # idx, cur, max
    delta = 0
    for tmp[0] in np.arange(0.1, 0.991, 0.01):
        tmp[1] = f1_score(y_train, np.array(train_preds)>tmp[0])
        if tmp[1] > tmp[2]:
            delta = tmp[0]
            tmp[2] = tmp[1]
    return delta , tmp[2]

def make_adddata( n_fold ):
    """ Note:
      If it is INSINCERE question,
      it should be regarded as insincere
      even if a sincere question continues.
      So we combile two questions to make new data.
      Feature and vectors make from two question's,
      it works faster than string processing.
    """
    np.random.seed( n_fold )
    if validation:
        y_noval = np.array(train_index)
    else:
        y_noval = np.arange(len(seq_train))
    tur_index = y_noval[y_train[y_noval] == 1]
    seq_true = seq_train[tur_index]
    fal_index = np.random.choice(y_noval[y_train[y_noval] != 1], len(seq_true), replace=True)
    seq_false = seq_train[fal_index]
    len_true = np.count_nonzero(seq_true, axis=1)
    len_false = np.count_nonzero(seq_false, axis=1)
    for i in range(len(seq_true)):
        fl = len_false[i]+len_true[i]
        if len_true[i] == 0:
            seq_false[i] = seq_true[i]
        elif fl >= train_maxlen:
            seq_false[i][0:len_true[i]] = seq_true[i][-len_true[i]:]
        else:
            seq_false[i][train_maxlen-fl:train_maxlen-fl+len_true[i]] = seq_true[i][-len_true[i]:]
    feature_new = feature_train[fal_index] + feature_train[tur_index]
    feature_new[:,10:] /= 2
    matrix_new = matrix_train[fal_index] + matrix_train[tur_index]
    return seq_false, feature_new, matrix_new

print('y_train==0',np.sum(y_train==0))
print('y_train==1',np.sum(y_train==1))

if validation:
    kf = KFold( n_splits=3, random_state=12, shuffle=True )
    total_score = []
    for fold_id, (train_index, test_index) in enumerate( kf.split( seq_train ) ):
        
        calc_idx = np.array_split(np.arange(len(test_index)),2)
        
        print("fat data")
        
        pred_total = np.zeros((len(test_index),))
        delta_total = 0
        with Pool(4) as p:
            result = p.map(make_adddata, np.arange(nth_fat_data))
        x_train = seq_train[train_index]
        f_train = feature_train[train_index]
        m_train = matrix_train[train_index]
        yy_train = y_train[train_index]
        for seq_new, feature_new, matrix_new in result:
            x_train = np.vstack( [x_train, seq_new] )
            f_train = np.vstack( [f_train, feature_new] )
            m_train = vstack( [m_train, matrix_new] )
            yy_train = np.hstack( [yy_train, np.ones((len(seq_new),)) ] )
            
        print("start train")
        
        fold_score = 0
        
        for est in range(n_estimaters):
            pred_test = run_one( x_train, f_train, m_train, yy_train, seq_train[test_index], feature_train[test_index], matrix_train[test_index], y_train[test_index] )
            pred_test = pred_test.reshape((-1,))
            delta, _ = bestThresshold(y_train[test_index[calc_idx[0]]], pred_test[calc_idx[0]])
            pred = (pred_test > delta).astype(int)
            score = f1_score(pred[calc_idx[1]], y_train[test_index[calc_idx[1]]])
            print("fold %d: %d# score of estimater: %f"%(fold_id,est,score))
            pred_total += pred_test
            delta_total += delta
            pred = (pred_total > delta_total).astype(int)
            score = f1_score(pred[calc_idx[1]], y_train[test_index[calc_idx[1]]])
            print("fold %d: %d# score of total: %f"%(fold_id,est,score))
            print(classification_report(pred[calc_idx[1]], y_train[test_index[calc_idx[1]]]))
            fold_score = score
        
        total_score.append(fold_score)
        print("final score: %f"%(np.mean(total_score),))
            
else:
    kf = KFold( n_splits=12, random_state=12, shuffle=True )
    pred_total = np.zeros((len(seq_test),))
    delta_total = 0
    for fold_id, (train_index, test_index) in enumerate( kf.split( seq_train ) ):
        
        if fold_id >= n_estimaters:
            break
        
        print("fat data")
        
        delta_total = 0
        with Pool(4) as p:
            result = p.map(make_adddata, np.arange(nth_fat_data))
        x_train = seq_train[train_index]
        f_train = feature_train[train_index]
        m_train = matrix_train[train_index]
        yy_train = y_train[train_index]
        for seq_new, feature_new, matrix_new in result:
            x_train = np.vstack( [x_train, seq_new] )
            f_train = np.vstack( [f_train, feature_new] )
            m_train = vstack( [m_train, matrix_new] )
            yy_train = np.hstack( [yy_train, np.ones((len(seq_new),)) ] )
            
        print("start train")

        x_test = seq_train[test_index]
        f_test = feature_train[test_index]
        m_test = matrix_train[test_index]
        yy_test = y_train[test_index]
        x_test = np.vstack( [x_test, seq_test] )
        f_test = np.vstack( [f_test, feature_test] )
        m_test = vstack( [m_test, matrix_test] )

        pred_test = run_one( x_train, f_train, m_train, yy_train, x_test, f_test, m_test )
        pred_test = pred_test.reshape((-1,))
        delta, _ = bestThresshold(yy_test, pred_test[:len(yy_test)])
        pred_total += pred_test[len(yy_test):]
        delta_total += delta
        
    pred_submit = (pred_total > delta_total).astype(int)
    pd.DataFrame({'qid':ids_test,'prediction':pred_submit}).to_csv('submission.csv', index=False)


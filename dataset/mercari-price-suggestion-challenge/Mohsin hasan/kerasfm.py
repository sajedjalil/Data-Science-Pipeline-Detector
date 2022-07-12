# Same procdessing as NNvnet1001
#Import libraries
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ["THEANO_FLAGS"] = "floatX=float32,device=cpu"
import copy
import numpy as np
np.random.seed(786)
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
import sys
from contextlib import closing
import re
import unicodedata
import string
import math
import gc
import time
#from time import time
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import itertools

from tqdm import tqdm
tqdm.pandas(tqdm)

from fastcache import clru_cache as lru_cache
import threading
import multiprocessing
from multiprocessing import Pool, cpu_count
num_partitions = 8
num_cores = 4

import wordbatch
from  wordbatch.extractors import WordSeq, WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL, NN_ReLU_H1, NN_ReLU_H2

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, RobustScaler, MaxAbsScaler, QuantileTransformer, OneHotEncoder
from sklearn import metrics
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin

from scipy.sparse.csr import csr_matrix
from scipy.sparse import hstack
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Reshape, concatenate, Dense, GlobalAveragePooling1D, BatchNormalization, Flatten, Dropout, PReLU, LeakyReLU, Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, GRU, LSTM, merge, Dot, Add, Lambda, Multiply
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Layer
from keras import initializers, regularizers, constraints
from keras.regularizers import l2 as l2_reg
#from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam, RMSprop, Nadam
import keras.backend as K
import tensorflow as tf

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_sklearn = make_scorer(rmse, greater_is_better=False)
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

stop_words = ['a', 'an', 'this', 'is', 'the', 'of', 'for']

def unicodeToAscii(series):
    return  series.apply(lambda s: unicodedata.normalize('NFKC', str(s)))

def multiple_replace(text, adict):
    rx = re.compile('|'.join(map(re.escape, adict)))
    def one_xlat(match):
        return adict[match.group(0)]
    return rx.sub(one_xlat, text)

# Lowercase, trim, and remove non-letter characters
def normalizeString(series):
    series = unicodeToAscii(series)

    replace_dict = {r" one ": " 1 ",
                    r" two ": r" 2 ",
                    r" three ": " 3 ",
                    r" four " : " 4 ", 
                    r" five " : " 5 ",
                    r" six "  : " 6 ", 
                    r" seven ": " 7 ",   
                    r"lily jade": "lilyjade", 
                    r"rae dunn cookies": "raedunncookie",
                    r"rae dunn cookie": "raedunncookie",
                    r"hatchimals" : "hatchimal", 
                    r"virtual reality" : "vr", 
                    r" vs " : " victorias secret ",
                    r" mk " : " michael kors ", 
                    r" victoria secret " : " victorias secret ",
                    r"google home" : "googlehome", 
                    r"16 gb" : "16gb ", 
                    r"256 gb" : "256gb ",
                    r"32 gb" : "32gb ", 
                    r"14k gold" : '14kgold', 
                    r"14 gold" : '14kgold',
                    r"14 k gold" : '14kgold',
                    r"lululemon bags" : 'lululemonbags',
                    r"controller skin" : 'controllerskin',
                    r"watch box" : 'watchbox',
                    r"blaze band" : 'blazeband', 
                    r"vault boy" : 'vaultboy',
                    r"lash boost" : 'lashboost', 
                    r"64 g " : '64gb ',
                    r"32 g " : '32gb',
                    r"go pro hero" : 'goprohero', 
                    r"gopro hero" : 'goprohero', 
                    r"nmd r " : 'nmdr ', 
                    r"nmd r1 " : 'nmdr ', 
                    r"nmds r " : 'nmdr ',  
                    r"private sale" : 'privatesale', 
                    r"vutton" : 'vuitton',
                    r"louis vuitton eva" : 'louisvuittoneva',
                    r"apple watch" : 'applewatch',
                   r"No description yet": "missing"}
    rx = re.compile('|'.join(map(re.escape, replace_dict)))
    def one_xlat(match):
        return replace_dict[match.group(0)]
    
    series = series.str.lower()
    series = series.str.replace(r"\'", "")
    series = series.str.replace(r"\-", "")
    series = series.str.replace(r"[^0-9a-zA-Z]+", " ")
    series = series.str.replace(r"(?=\w{1,2})iphone ", "iphone")
    series = series.str.replace(r"(?=\w{1,2})galaxy ", "galaxy")
    series = series.str.replace(rx, one_xlat)

    return series

###########################

def token_generator(texts):
    for text in texts:
        yield str(text).split()

class Tokenizer:
    def __init__(self, max_features=20000, tokenizer=str.split):
        self.max_features = max_features
        self.tokenizer = tokenizer
        self.doc_freq = None
        self.vocab = None
        self.vocab_idx = None
        self.max_len = None

    def fit_transform(self, texts):
        tokenized = []
        n = len(texts)



        tokenized = token_generator(texts)
        doc_freq = Counter(itertools.chain.from_iterable(tokenized))
        
        vocab = [t[0] for t in doc_freq.most_common(self.max_features)]
        vocab_idx = {w: (i + 1) for (i, w) in enumerate(vocab)}
        #doc_freq = [doc_freq[t] for t in vocab]

        #self.doc_freq = doc_freq
        self.vocab = vocab
        self.vocab_idx = vocab_idx

        max_len = 0
        result_list = []
        tokenized = token_generator(texts)
        for text in tokenized:
            text = self.text_to_idx(text)
            max_len = max(max_len, len(text))
            result_list.append(text)

        self.max_len = max_len
        result = np.zeros(shape=(n, max_len), dtype=np.int32)
        for i in range(n):
            text = result_list[i]
            result[i, :len(text)] = text

        return result    

    def text_to_idx(self, tokenized):
        return [self.vocab_idx[t] for t in tokenized if t in self.vocab_idx]

    def transform(self, texts):
        n = len(texts)
        result = np.zeros(shape=(n, self.max_len), dtype=np.int32)

        for i in range(n):
            text = self.tokenizer(texts[i])
            text = self.text_to_idx(text)[:self.max_len]
            result[i, :len(text)] = text

        return result
    
    def vocabulary_size(self):
        return len(self.vocab) + 1

def grams2(src_words):
    return list(zip(src_words,src_words[1:]))

def tokenizer_2gram(text):
    return grams2(str(text).split())

def ngrams(n, f, prune_after=10000):
    counter = collections.Counter()
    deque = collections.deque(maxlen=n)
    for i, line in enumerate(f):
        deque.clear()
        words =  str(line).split() 
        deque.extend(words[:n-1])
        for word in words[n-1:]:
            #if word in vocab:
            deque.append(word)
            ngram = tuple(str(w) for w in deque)
            if i < prune_after or ngram in counter:
                counter[ngram] += 1
    return counter



def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    with closing(Pool(num_cores)) as pool:
        df = pd.concat(pool.map(func, df_split))
    return df

def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['category_name'].isin(pop_category1), 'category_name'] = 'missing'

    
def generate_features(data):
    data = data.copy()

    data.fillna(-1, inplace=True)
    data['cat1'] = data.category_name.str.split('/').str.get(0).astype('category')
    data['cat2'] = data.category_name.str.split('/').str.get(1).astype('category')
    data['cat3'] = data.category_name.str.split('/').str.get(2).astype('category')

    print("Getting word/char len features")
    data['desc_words'] = (data.item_description.str.count('\s+') + 1).fillna(0).astype(int)
    data['desc_chars'] = data.item_description.str.len().fillna(0).astype(int)
    data['name_words'] = (data.name.str.count('\s+') + 1).fillna(0).astype(int)
    data['name_chars'] = data.name.str.len().fillna(0).astype(int)

    print("Get count features")
    data['brand_counts'] = data.brand_name.map(data["brand_name"].value_counts()).fillna(0).astype(int)

    data['cat_counts'] = data.category_name.map(data["category_name"].value_counts()).fillna(0).astype(int)

    data['cat1_counts'] = data.cat1.map(data["cat1"].value_counts()).fillna(0).astype(int)

    data['cat2_counts'] = data.cat2.map(data["cat2"].value_counts()).fillna(0).astype(int)

    data['cat3_counts'] = data.cat3.map(data["cat3"].value_counts()).fillna(0).astype(int)

    data["brand_cat"] = (data["brand_name"].astype(str) + ' ' + data["category_name"].astype(str)).astype('category')
    data["cat_cond"] = (data["category_name"].astype(str) + ' ' + data["item_condition_id"].astype(str)).astype('category')
    data["brand_cond"] = (data["brand_name"].astype(str) + ' ' + data["item_condition_id"].astype(str)).astype('category')
    data["category_shipping"] = (data["category_name"].astype(str) + ' ' + data["shipping"].astype(str)).astype('category')

    data['brand_cat_counts'] = data.brand_cat.map(data["brand_cat"].value_counts()).fillna(0).astype(int)

    num_cols =  ["desc_words", "desc_chars", "name_words", "name_chars", 
                 "brand_counts", "cat_counts", "cat1_counts", 
               "cat2_counts", "cat3_counts", "brand_cat_counts"]
    return data


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))),
                        (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim
        
def grams2(src_words):
    return list(zip(src_words,src_words[1:]))

def ngrams3(n, f, prune_after=10000):
    #counter = collections.Counter()
    #deque = collections.deque(maxlen=n)
    for i, line in enumerate(f):
        #deque.clear()
        words =  str(line).split() 
        #deque.extend(words[:n-1])
        #for word in words[n-1:]:
            #if word in vocab:
        #    deque.append(word)
        #    ngram = tuple(str(w) for w in deque)
        yield grams2(words)
    
class NgramTokenizer:
    def __init__(self, max_features=20000, ngram=2):
        self.max_features = max_features
        self.doc_freq = None
        self.vocab = None
        self.vocab_idx = None
        self.max_len = None
        self.ngram = ngram

    def fit_transform(self, texts):
        tokenized = []
        #doc_freq = Counter()
        n = len(texts)
        tokenized = ngrams3(self.ngram, texts, prune_after=3000000)
        doc_freq = Counter(itertools.chain.from_iterable(tokenized))
        
        #vocab = sorted([t for (t, c) in doc_freq.items() if c >= self.min_df])
        vocab = [t[0] for t in doc_freq.most_common(self.max_features)]
        vocab_idx = {w: (i + 1) for (i, w) in enumerate(vocab)}
        #doc_freq = [doc_freq[t] for t in vocab]
        
        #self.doc_freq = doc_freq
        self.vocab = vocab
        self.vocab_idx = vocab_idx
        print("Vocab building done")
        max_len = 0
        result_list = []
        tokenized = ngrams3(self.ngram, texts, prune_after=3000000)
        for text in tokenized:
            text = self.text_to_idx(text)
            max_len = max(max_len, len(text))
            result_list.append(text)

        self.max_len = max_len
        result = np.zeros(shape=(n, max_len), dtype=np.int32)
        for i in range(n):
            text = result_list[i]
            result[i, :len(text)] = text

        return result 
    
    def text_to_idx(self, tokenized):
        return [self.vocab_idx[t] for t in tokenized if t in self.vocab_idx]

    def transform(self, texts):
        n = len(texts)
        result = np.zeros(shape=(n, self.max_len), dtype=np.int32)

        for i in range(n):
            text = grams2(str(texts[i]).split())
            text = self.text_to_idx(text)[:self.max_len]
            result[i, :len(text)] = text

        return result
    
    def vocabulary_size(self):
        return len(self.vocab) + 1
        

start_time = time.time()
in_path = "../input/"
train_data = pd.read_table(os.path.join(in_path, 'train.tsv'))
test_data  = pd.read_table(os.path.join(in_path, 'test.tsv'))
train_data = train_data.loc[(train_data.price > 2)].reset_index(drop=True)
train_rows = len(train_data)
data = pd.concat([train_data, test_data], ignore_index=True)

print(train_data.shape, test_data.shape)
print("Reading data completed in {}".format(time.time()- start_time))

gc.collect()
data = generate_features(data)
print(data.shape)
print("Additional feature generation completed in {}".format(time.time()- start_time))

for col in ['name', 'item_description']:
    train_data[col] = parallelize_dataframe(train_data[col], normalizeString)
    test_data[col] = parallelize_dataframe(test_data[col], normalizeString)
print(test_data.name.sample(1))
print("Text normalization completed in {}".format(time.time()- start_time))

gc.collect()
print('processing categories...')
X_cats = {}
X_test_cats = {}
for col in ["brand_name", "category_name", "cat1", "cat2", "cat3", 
            "brand_cat", "category_shipping", "item_condition_id", "cat_cond", "brand_cond"]:
    data[col] = data[col].astype('category').cat.codes + 1
    X_cats[col] = data.loc[:train_rows-1, col].astype(np.int32).values.reshape(-1,1)
    X_test_cats[col] = data.loc[train_rows:, col].astype(np.int32).values.reshape(-1,1)

X_conts = []
X_test_conts = []
for col in ['shipping', 'desc_words', 'desc_chars',
            'name_words', 'name_chars',  
            'brand_counts', 'cat_counts', 'cat1_counts', 
            'cat2_counts', 'cat3_counts', 
           ]:
    print(col)
    X_conts.append(data.loc[:train_rows-1, col].replace('missing', 0).astype(np.float32).values)
    X_test_conts.append(data.loc[train_rows:, col].replace('missing', 0).astype(np.float32).values)

X_conts = np.array(X_conts).T
X_test_conts = np.array(X_test_conts).T

print(X_conts.shape, X_test_conts.shape)
print("Categoricals data preparation completed in {}".format(time.time()- start_time))


name_num_col = 7
desc_num_col = 70
desc2_num_col = 20

def name_proc():
    print('processing title...')
    name_tok = Tokenizer(40000)
    X_name = name_tok.fit_transform(train_data["name"])
    print(X_name.shape)
    X_name = X_name[:, :name_num_col]
    X_test_name = name_tok.transform(test_data["name"])
    X_test_name = X_test_name[:, :name_num_col]
    name_voc_size = name_tok.vocabulary_size()
    print(name_voc_size)
    return X_name, X_test_name, name_tok


def desc_proc():
    print('processing description...')
    desc_tok = Tokenizer(50000)
    X_desc = desc_tok.fit_transform(train_data["item_description"])
    X_desc = X_desc[:, :desc_num_col]

    X_test_desc = desc_tok.transform(test_data["item_description"])
    X_test_desc = X_test_desc[:, :desc_num_col]
    desc_voc_size = desc_tok.vocabulary_size()
    print(desc_voc_size)

    return X_desc, X_test_desc, desc_tok

with Pool(2) as pool:
    res1 = pool.apply_async(name_proc, args=())
    res2 = pool.apply_async(desc_proc, args=())

    X_name, X_test_name, name_tok = res1.get()
    X_desc, X_test_desc, desc_tok = res2.get()



gc.collect()

name_voc_size = name_tok.vocabulary_size()

desc_voc_size = desc_tok.vocabulary_size()
print("Name and description data preparation completed in {}".format(time.time()- start_time))


desc2_num_col = 20
name2_num_col = 6
itemname_num_col = 10
cat_seq_len = 7
train_data["catnorm"] = normalizeString(train_data["category_name"].fillna("missing")) 
test_data["catnorm"] = normalizeString(test_data["category_name"].fillna("missing")) 
def desc2_proc():
    print('processing description2...')
    desc2_tok = NgramTokenizer(40000)
    X_desc2 = desc2_tok.fit_transform(train_data["item_description"])
    X_desc2 = X_desc2[:, :desc2_num_col]

    X_test_desc2 = desc2_tok.transform(test_data["item_description"])
    X_test_desc2 = X_test_desc2[:, :desc2_num_col]
    desc2_voc_size = desc2_tok.vocabulary_size()
    print(desc2_voc_size)

    return X_desc2, X_test_desc2, desc2_tok

def name2_proc():
    print('processing title2...')
    name2_tok = NgramTokenizer(40000)
    X_name2 = name2_tok.fit_transform(train_data["name"])
    print(X_name2.shape)
    X_name2 = X_name2[:, :name2_num_col]
    X_test_name2 = name2_tok.transform(test_data["name"])
    X_test_name2 = X_test_name2[:, :name2_num_col]
    name2_voc_size = name2_tok.vocabulary_size()
    print(name2_voc_size)
    return X_name2, X_test_name2, name2_tok

def itemname_proc():
    print('procecssing itemname...')
    X_itemname = name_tok.transform(train_data['item_description'])
    X_test_itemname = name_tok.transform(test_data['item_description'])
    X_itemname = X_itemname[:, :itemname_num_col]
    X_test_itemname = X_test_itemname[:, :itemname_num_col]
    name_voc_size = name_tok.vocabulary_size()

    return X_itemname, X_test_itemname, name_tok

def catnorm_proc():
    cat_tok = Tokenizer(1000)
    X_catnorm = cat_tok.fit_transform(train_data["catnorm"])
    X_test_catnorm = cat_tok.transform(train_data["catnorm"])

    X_catnorm = X_catnorm[:, :cat_seq_len]
    X_test_catnorm = X_test_catnorm[:, :cat_seq_len]

    X_namecat = cat_tok.transform(train_data['name'].astype(str))
    X_test_namecat = cat_tok.transform(test_data['name'].astype(str))

    X_desccat = cat_tok.transform(train_data['item_description'].astype(str))
    X_test_desccat = cat_tok.transform(test_data['item_description'].astype(str))

    return X_catnorm, X_test_catnorm, X_namecat, X_test_namecat, X_desccat, X_test_desccat, cat_tok


with Pool(2) as pool:
    res1 = pool.apply_async(name2_proc, args=())
    res2 = pool.apply_async(desc2_proc, args=())
    #res3 = pool.apply_async(itemname_proc, args=())
    #res4 = pool.apply_async(catnorm_proc, args=())
    #res4 = pool.apply_async(catname_proc, args=())
    X_name2, X_test_name2, name2_tok = res1.get()
    X_desc2, X_test_desc2, desc2_tok = res2.get()
    #X_itemname, X_test_itemname, name_tok = res3.get()
    #X_catnorm, X_test_catnorm, X_namecat, X_test_namecat, X_desccat, X_test_desccat, cat_tok = res4.get()
gc.collect()

name2_voc_size = name2_tok.vocabulary_size()

desc2_voc_size = desc2_tok.vocabulary_size()

#catnorm_voc_size = cat_tok.vocabulary_size()
print("Name and description 2 grams data preparation completed in {}".format(time.time()- start_time))
del name2_tok, desc2_tok


y = np.log1p(train_data.price).values.reshape(-1,1)
cvlist = list(KFold(40, random_state=1).split(X_desc))
std = np.std(y)
mean = np.mean(y)
ynorm = (y - mean)/std

name_seq_len = X_name.shape[1]
name_embeddings_dim = 50

itemname_embeddings_dim = 30

desc_seq_len = X_desc.shape[1]
desc_embeddings_dim = 50

desc2_seq_len = X_desc2.shape[1]
desc2_embeddings_dim = 30

name2_seq_len = X_name2.shape[1]
name2_embeddings_dim = 30

brand_voc_size = max(np.max(X_cats["brand_name"]), np.max(X_test_cats["brand_name"])) + 1
brand_embeddings_dim = 50

cat_voc_size = max(np.max(X_cats["category_name"]), np.max(X_test_cats["category_name"])) + 1
cat_embeddings_dim = 40

cat1_voc_size = max(np.max(X_cats["cat1"]), np.max(X_test_cats["cat1"])) + 1
cat1_embeddings_dim = 6

cat2_voc_size = max(np.max(X_cats["cat2"]), np.max(X_test_cats["cat2"])) + 1
cat2_embeddings_dim = 15

cat3_voc_size = max(np.max(X_cats["cat3"]), np.max(X_test_cats["cat3"])) + 1
cat3_embeddings_dim = 20

brandcat_voc_size = max(np.max(X_cats["brand_cat"]), np.max(X_test_cats["brand_cat"])) + 1
brandcat_embeddings_dim = 30

catship_voc_size = max(np.max(X_cats["category_shipping"]), np.max(X_test_cats["category_shipping"])) + 1
catship_embeddings_dim = 20

catcond_voc_size = max(np.max(X_cats["cat_cond"]), np.max(X_test_cats["cat_cond"])) + 1
catcond_embeddings_dim = 20

brandcond_voc_size = max(np.max(X_cats["brand_cond"]), np.max(X_test_cats["brand_cond"])) + 1
brandcond_embeddings_dim = 20

cond_voc_size = max(np.max(X_cats["item_condition_id"]), np.max(X_test_cats["item_condition_id"])) + 1
cond_embedding_size = 4

catnorm_embeddings_dim = 6


valid=True
l2_fm=0.1
def keras_ffm_model(seed, K, l2, l2_fm):
    #Get all input params

    name = Input(shape=(name_seq_len,))
    name2 = Input(shape=(name2_seq_len,))
    desc = Input(shape=(desc_seq_len,))
    desc2 = Input(shape=(desc2_seq_len,))
    name2 = Input(shape=(name2_seq_len,))
    itemname = Input(shape=(itemname_num_col,))
    brand = Input(shape=(1,))
    cat = Input(shape=(1,))
    cat1 = Input(shape=(1,))
    cat2 = Input(shape=(1,))
    cat3 = Input(shape=(1,))
    brandcat = Input(shape=(1,))
    catship = Input(shape=(1,))
    brandcond = Input(shape=(1,))
    catcond = Input(shape=(1,))
    cond = Input(shape=(1,))
    ship = Input(shape=(1,))
    catnorm = Input(shape=(cat_seq_len,))
    namecat = Input(shape=(cat_seq_len,))
    desccat = Input(shape=(cat_seq_len,))
    
    inputs = []
    flatten_layers = []
    #inputs = [name, desc, 
    #          desc2, name2, 
    #          itemname, brand, cat, cat1, cat2, cat3, brandcat, catship, brandcond, catcond, cond, conts,
    #         catnorm, namecat, desccat]


 
    columns = [('name',name_voc_size, name_seq_len, name, 0), 
               ('name2',name2_voc_size, name2_seq_len, name2, 0.0000),
               ('desc', desc_voc_size, desc_seq_len, desc, 0.00),
               ('desc2', desc2_voc_size, desc2_seq_len, desc2, 0.0),
               ('cat', cat_voc_size, 1, cat, 0), 
               #('cat1', cat1_voc_size, 1, cat1, 0),
               #('cat2', cat2_voc_size, 1, cat2, 0),
               #('cat3', cat3_voc_size, 1, cat3, 0),
               ('brand', brand_voc_size, 1, brand, 0), 
               ('cond', cond_voc_size, 1, cond, 0), ('ship', 2, 1, ship, 0),
               #('catship', catship_voc_size, 1, catship, 0),
              ]
    
    
    #embed_name = Embedding(name_voc_size, name_embeddings_dim,  embeddings_initializer='he_uniform')
    #name = embed_name(name)
    #name2 = Embedding(name2_voc_size, name2_embeddings_dim, embeddings_initializer='he_uniform')(name2)
    #desc = Embedding(desc_voc_size, desc_embeddings_dim, embeddings_initializer='he_uniform')(desc)
    #desc2 = Embedding(desc2_voc_size, desc2_embeddings_dim, embeddings_initializer='he_uniform')(desc2)
    #itemname = Embedding(name_voc_size, itemname_embeddings_dim, embeddings_initializer='he_uniform')(itemname)
    #catnorm = Embedding(catnorm_voc_size, catnorm_embeddings_dim, embeddings_initializer='he_uniform')(catnorm)
    #namecat = Embedding(catnorm_voc_size, catnorm_embeddings_dim, embeddings_initializer='he_uniform')(namecat)
    #desccat = Embedding(catnorm_voc_size, catnorm_embeddings_dim, embeddings_initializer='he_uniform')(desccat)
    
    #brand = Embedding(brand_voc_size, brand_embeddings_dim, embeddings_initializer='he_uniform')(brand)
    #cat = Embedding(cat_voc_size, cat_embeddings_dim, embeddings_initializer='he_uniform')(cat)
    #cat1 = Embedding(cat1_voc_size, cat1_embeddings_dim, embeddings_initializer='he_uniform')(cat1)
    #cat2 = Embedding(cat2_voc_size, cat2_embeddings_dim, embeddings_initializer='he_uniform')(cat2)
    #cat3 = Embedding(cat3_voc_size, cat3_embeddings_dim, embeddings_initializer='he_uniform')(cat3)
    #brandcat = Embedding(brandcat_voc_size, brandcat_embeddings_dim, embeddings_initializer='he_uniform')(brandcat)
    #brandcond = Embedding(brandcond_voc_size, brandcond_embeddings_dim, embeddings_initializer='he_uniform')(brandcond)
    #catcond = Embedding(catcond_voc_size, catcond_embeddings_dim, embeddings_initializer='he_uniform')(catcond)
    #catship = Embedding(catship_voc_size, catship_embeddings_dim, embeddings_initializer='he_uniform')(catship)
    #cond = Embedding(cond_voc_size, cond_embedding_size, embeddings_initializer='he_uniform')(cond)
    
    for c, num_c, seq_len, inputs_c, reg in columns:
        #num_c = max_features[c]

        embed_c = Embedding(
                        num_c,
                        K,
                        input_length=seq_len,
                        name = 'embed_%s'%c,
                        W_regularizer=l2_reg(reg)
                        )(inputs_c)
        if seq_len > 1:
            flatten_c = GlobalAveragePooling1D()(embed_c)
        else:
            flatten_c = Flatten()(embed_c)

        inputs.append(inputs_c)
        flatten_layers.append(flatten_c)
    
    fm_layers = []
    for emb1,emb2 in itertools.combinations(flatten_layers, 2):
        dot_layer = merge([emb1,emb2],mode='dot',dot_axes=1)
        fm_layers.append(dot_layer)
    
    for i, (c, num_c, seq_len, input_c, reg) in enumerate(columns):
        embed_c = Embedding(
                        num_c,
                        1,
                        input_length=seq_len,
                        name = 'linear_%s'%c,
                        W_regularizer=l2_reg(l2)
                        )(inputs[i])

        if seq_len > 1:
            flatten_c = GlobalAveragePooling1D()(embed_c)
        else:
            flatten_c = Flatten()(embed_c)

        fm_layers.append(flatten_c)
    
    out = merge(fm_layers,mode='sum')
    #out = Dropout(0.1)(out)
    #out = BatchNormalization(momentum=0.99, scale=False, center=False)(out)
    out = Dense(250, activation='selu', kernel_initializer='he_normal')(out)
    #out = Dense(150, activation='tanh', kernel_initializer='he_normal')(out)
    #out = PReLU()(out)
    
    print(out.shape)

    out = Dense(1, kernel_initializer='normal')(out)

    model = Model(inputs, out)
    opt = Adam(lr=0.005, decay=0.00, clipnorm=1.0)
    model.compile(optimizer=opt, loss="mse" )
    return model
    
seed=0
batchsize=2000
epochs=2
np.random.seed(seed)
tf.set_random_seed(seed)

model = keras_ffm_model(seed, 60, 0.0, 0.0)

train_idx, val_idx= cvlist[seed]

X = [X_name, X_name2, X_desc, X_desc2, 
    X_cats["category_name"], 
     #X_cats["cat1"], 
     #X_cats["cat2"],  X_cats["cat3"],
     X_cats["brand_name"], X_cats["item_condition_id"], X_conts[:,0].reshape(-1,1),
    #X_cats["category_shipping"]
    ]
    
X_test = [X_test_name, X_test_name2, X_test_desc, X_test_desc2, 
    X_test_cats["category_name"], 
     #X_cats["cat1"], 
     #X_cats["cat2"],  X_cats["cat3"],
     X_test_cats["brand_name"], X_test_cats["item_condition_id"], X_test_conts[:,0].reshape(-1,1),
    #X_cats["category_shipping"]
    ]

X_tr = [x[train_idx] for x in X]
X_val = [x[val_idx] for x in X]

lr1, lr2, lr3 = [0.0045, 0.001, 0.0003]
lrs = [lr1, lr2, lr3, 0.001, 0.001, 0.001]
def schedule(epoch):
    return lrs[epoch]

lr_schedule = LearningRateScheduler(schedule)
#val_store = TestCallback(X_val, X_test)
gc.collect()
if valid:
    model.fit(X_tr, y[train_idx], batch_size=batchsize, epochs=epochs,
                               verbose=1,
                              validation_data=(X_val, y[val_idx]), shuffle=True,
                              callbacks= [lr_schedule]
         )
    y_val = y[val_idx,0]
    y_pred = model.predict(X_val, batch_size=50000)[:, 0]
    print(np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
else:
    model.fit(X, y, batch_size=batchsize, epochs=epochs,
                               verbose=0,
                               shuffle=True,
                              callbacks= [lr_schedule]
         )
         
y_test_pred = model.predict(X_test)[:, 0]

print("Write out submission")
submission  = test_data[['test_id']]
submission['price'] = np.expm1(y_test_pred)
submission.price = submission.price.clip(3, 2000)
submission.to_csv("embedding_nn_v2.csv", index=False)
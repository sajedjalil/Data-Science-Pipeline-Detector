import gc
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.preprocessing import LabelEncoder, Normalizer

from sklearn.model_selection import train_test_split

import sys
import os
import random
import numpy as np
import tensorflow as tf
# os.environ['PYTHONHASHSEED'] = '10000'
# np.random.seed(10001)
# random.seed(10002)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=5, inter_op_parallelism_threads=1)
from keras import backend as K
# tf.set_random_seed(10003)
K.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))
#Add https://www.kaggle.com/anttip/wordbatch to your kernel Data Sources, 
#until Kaggle admins fix the wordbatch pip package installation

from nltk.corpus import stopwords
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K

NUM_BRANDS = 4500
NUM_CATEGORIES = 1250

develop = False
# develop= True

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("missing", "missing", "missing")


def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='No description yet', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'


def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')

# get name and description lengths
def wordCount(text):
    try:
        if text == 'No description yet':
            return 0
        else:
            text = text.lower()
            words = [w for w in text.split(" ")]
            return len(words)
    except: 
        return 0

def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])

def normalize_dataset_text(dataset):
    dataset['item_description'] = dataset['item_description'].apply(lambda x: normalize_text(x))
    dataset['brand_name'] = dataset['brand_name'].apply(lambda x: normalize_text(x))

def delete_unseen(dataset):
    dataset.loc[~dataset['brand_name'].isin(all_brand), 'brand_name'] = 'missing'
    dataset.loc[~dataset['general_cat'].isin(all_general_cat), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(all_subcat_1), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(all_subcat_2), 'subcat_2'] = 'missing'

def text_length_feature(dataset, train = True):
    if train:
        dataset['desc_len'] = dataset['item_description'].apply(lambda x: wordCount(x))
        dataset['name_len'] = dataset['name'].apply(lambda x: wordCount(x))
        dataset[['desc_len', 'name_len']] = desc_normalizer.fit_transform(dataset[['desc_len', 'name_len']])
    else:
        dataset['desc_len'] = dataset['item_description'].apply(lambda x: wordCount(x))
        dataset['name_len'] = dataset['name'].apply(lambda x: wordCount(x))
        dataset[['desc_len', 'name_len']] = desc_normalizer.transform(dataset[['desc_len', 'name_len']])

start_time = time.time()
from time import gmtime, strftime
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

train = pd.read_table('../input/train.tsv', engine='c')
if develop:
    train, dev = train_test_split(train, test_size=0.025, random_state=200)
    dev_y = np.log1p(dev["price"])
print('[{}] Finished to load data'.format(time.time() - start_time))
print('Train shape: ', train.shape)
#dftt = train[(train.price < 1.0)]
train = train.drop(train[(train.price < 1.0)].index)
#del dftt['price']
# print(nrow_train, nrow_test)
train_y = np.log1p(train["price"])

train['general_cat'], train['subcat_1'], train['subcat_2'] = \
    zip(*train['category_name'].apply(lambda x: split_cat(x)))
train.drop('category_name', axis=1, inplace=True)
print('[{}] Split categories completed.'.format(time.time() - start_time))

handle_missing_inplace(train)
print('[{}] Handle missing completed.'.format(time.time() - start_time))

cutting(train)
print('[{}] Cut completed.'.format(time.time() - start_time))

to_categorical(train)
print('[{}] Convert categorical completed'.format(time.time() - start_time))

desc_normalizer = Normalizer()
name_normalizer = Normalizer()
text_length_feature(train)
print('[{}] Calculate length features'.format(time.time() - start_time))

normalize_dataset_text(train)
print('[{}] Normalization text'.format(time.time() - start_time))

## get all categorical in train and replace missing value
all_brand = set(train["brand_name"].values)
all_general_cat = set(train["general_cat"].values)
all_subcat_1 = set(train["subcat_1"].values)
all_subcat_2 = set(train["subcat_2"].values)

le_brand = LabelEncoder()
le_general_cat = LabelEncoder()
le_subcat_1 = LabelEncoder()
le_subcat_2 = LabelEncoder()

le_brand.fit(train['brand_name'])
train['encoded_brand_name'] = le_brand.transform(train['brand_name'])

le_general_cat.fit(train['general_cat'])
train['encoded_general_cat'] = le_general_cat.transform(train['general_cat'])

le_subcat_1.fit(train['subcat_1'])
train['encoded_subcat_1'] = le_subcat_1.transform(train['subcat_1'])

le_subcat_2.fit(train['subcat_2'])
train['encoded_subcat_2'] = le_subcat_2.transform(train['subcat_2'])

print("Tokenizing item description")
tok_desc = Tokenizer()
tok_desc.fit_on_texts(train["item_description"].values)

print("Tokenizing name")
tok_name = Tokenizer()
tok_name.fit_on_texts(train["name"].values)

print("Transforming text to sequences...")
train['seq_item_description'] = tok_desc.texts_to_sequences(train["item_description"].values)
train['seq_name'] = tok_name.texts_to_sequences(train["name"].values)

## padding max length
MAX_NAME_SEQ = 15 #17
MAX_ITEM_DESC_SEQ = 50 #269

## embedding max length
MAX_DESC_TEXT = len(tok_desc.word_index) + 1
MAX_NAME_TEXT = len(tok_name.word_index) + 1
MAX_BRAND = len(le_brand.classes_)
MAX_GENCAT = len(le_general_cat.classes_)
MAX_SUBCAT_1 = len(le_subcat_1.classes_)
MAX_SUBCAT_2 = len(le_subcat_2.classes_)
MAX_CONDITION = max(train.item_condition_id) + 1

def get_rnn_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        'brand_name': np.array(dataset.encoded_brand_name),
        'item_condition': np.array(dataset.item_condition_id),
        'num_vars': np.array(dataset[["shipping"]]),
        'desc_len': np.array(dataset[["desc_len"]]),
        'name_len': np.array(dataset[["name_len"]]),
        'general_cat': np.array(dataset.encoded_general_cat),
        'subcat_1': np.array(dataset.encoded_subcat_1),
        'subcat_2': np.array(dataset.encoded_subcat_2),
    }
    return X

train_X = get_rnn_data(train)

## RNN Model
np.random.seed(123)

def rnn_model(lr=0.001, decay=0.0):
    # Inputs
    name = Input(shape=[train_X["name"].shape[1]], name="name")
    item_desc = Input(shape=[train_X["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    general_cat = Input(shape=[1], name="general_cat")
    subcat_1 = Input(shape=[1], name="subcat_1")
    subcat_2 = Input(shape=[1], name="subcat_2")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[train_X["num_vars"].shape[1]], name="num_vars")
    desc_len = Input(shape=[1], name="desc_len")
    name_len = Input(shape=[1], name="name_len")

    # Embeddings layers (adjust outputs to help model)
    emb_name = Embedding(MAX_NAME_TEXT, 20)(name)
    emb_item_desc = Dropout(0.05) (Embedding(MAX_DESC_TEXT, 50)(item_desc))
    emb_brand_name = Embedding(MAX_BRAND, 20)(brand_name)
    emb_general_cat = Embedding(MAX_GENCAT, 5)(general_cat)
    emb_subcat_1 = Embedding(MAX_SUBCAT_1, 10)(subcat_1)
    emb_subcat_2 = Embedding(MAX_SUBCAT_2, 15)(subcat_2)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
#     emb_shipping = Embedding(2, 5)(num_vars)
    

    # rnn layers (GRUs are faster than LSTMs and speed is important here)
    rnn_layer1 = GRU(16) (emb_item_desc)
    rnn_layer2 = GRU(8) (emb_name)
    # main layers
    main_l = concatenate([
        Flatten() (emb_brand_name)
        , Flatten() (emb_item_condition)
        , Flatten() (emb_general_cat)
        , Flatten() (emb_subcat_1)
        , Flatten() (emb_subcat_2)
#         , Flatten() (emb_shipping)
        , num_vars
        , rnn_layer1
        , rnn_layer2
        , desc_len
        , name_len
    ])
    # (incressing the nodes or adding layers does not effect the time quite as much as the rnn layers)
    main_l = Dropout(0.05)(Dense(512,kernel_initializer='normal',activation='relu') (main_l))
#     main_l = Dropout(0.05)(Dense(128,kernel_initializer='normal',activation='relu') (main_l))
#     main_l = Dropout(0.1)(Dense(256,kernel_initializer='normal',activation='relu') (main_l))
    main_l = Dropout(0.05)(Dense(96,kernel_initializer='normal',activation='relu') (main_l))
#     main_l = Dense(512,kernel_initializer='normal',activation='relu') (main_l)
#     main_l = Dense(64,kernel_initializer='normal',activation='relu') (main_l)

    # the output layer.
    output = Dense(1, activation="linear") (main_l)
    
    model = Model([name, item_desc, brand_name,
                   general_cat, subcat_1, subcat_2,
                   item_condition, num_vars, desc_len, name_len], output)

    optimizer = Adam(lr=lr, decay=decay)
    # (mean squared error loss function works as well as custom functions)  
    model.compile(loss = 'mse', optimizer = optimizer)

    return model

# Set hyper parameters for the model.
BATCH_SIZE = 512 * 3
epochs = 3

# Calculate learning rate decay.
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(train_X['name']) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.007, 0.003
lr_decay = exp_decay(lr_init, lr_fin, steps)

# Create model and fit it with training dataset.
model = rnn_model(lr=lr_init, decay=lr_decay)
model.fit(train_X, train_y, epochs=epochs, batch_size=BATCH_SIZE, verbose=2)


## Wordbatch
min_df = 1
import wordbatch

from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer

## name
wb_name = wordbatch.WordBatch(normalize_text, extractor=(WordBag, 
                                                            {"hash_ngrams": 2, 
                                                             "hash_ngrams_weights": [1.5, 1.0],
                                                             "hash_size": 2 ** 26, 
                                                             "norm": None, 
                                                             "tf": 'binary',
                                                             "idf": None}), procs=8)
wb_name.dictionary_freeze= True

## category
vec_gen = CountVectorizer()
vec_1 = CountVectorizer()
vec_2 = CountVectorizer()

## description
wb_desc = wordbatch.WordBatch(normalize_text, extractor=(WordBag, 
                                                    {"hash_ngrams": 2, 
                                                     "hash_ngrams_weights": [1.0, 1.0],
                                                     "hash_size": 2 ** 26, 
                                                     "norm": "l2", "tf": 1.0,
                                                      "idf": None}), procs=8)
wb_desc.dictionary_freeze= True

## brand name
lb = LabelBinarizer(sparse_output=True)

def feature_extract(dataset, train = True, name_mask = None, desc_mask = None, all_mask = None):
    if train:
        ## name
        X_name = wb_name.fit_transform(dataset['name'])
        name_mask = np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)
        X_name = X_name[:, name_mask]
        
        ## category
        X_category1 = vec_gen.fit_transform(dataset['general_cat'])
        X_category2 = vec_1.fit_transform(dataset['subcat_1'])
        X_category3 = vec_2.fit_transform(dataset['subcat_2'])
        
        ## description
        X_description = wb_desc.fit_transform(dataset['item_description'])
        desc_mask = np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)
        X_description = X_description[:, desc_mask]
                
        ## brand name
        X_brand = lb.fit_transform(dataset['brand_name'])
        
        X_dummies = csr_matrix(pd.get_dummies(dataset[['item_condition_id', 'shipping']],
                                      sparse=True).values)
        sparse_merge = hstack((X_dummies, X_description, X_brand, 
                       X_category1, X_category2, X_category3, 
                       X_name)).tocsr()
        print('[{}] FM Train Preprocessing Complete'.format(time.time() - start_time))
        all_mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
        sparse_merge = sparse_merge[:, all_mask]
        gc.collect()
    else:
        ## name
        X_name = wb_name.transform(dataset['name'])
        X_name = X_name[:, name_mask]
        
        ## category
        X_category1 = vec_gen.transform(dataset['general_cat'])
        X_category2 = vec_1.transform(dataset['subcat_1'])
        X_category3 = vec_2.transform(dataset['subcat_2'])
        
        ## description
        X_description = wb_desc.transform(dataset['item_description'])
        X_description = X_description[:, desc_mask]
        
        ## brand name
        X_brand = lb.transform(dataset['brand_name'])
        
        X_dummies = csr_matrix(pd.get_dummies(dataset[['item_condition_id', 'shipping']],
                                      sparse=True).values)
        sparse_merge = hstack((X_dummies, X_description, X_brand, 
               X_category1, X_category2, X_category3, 
               X_name)).tocsr()
        print('[{}] FM Test/Dev Preprocessing Complete'.format(time.time() - start_time))
        sparse_merge = sparse_merge[:, all_mask]
        gc.collect()
    return(sparse_merge, name_mask, desc_mask, all_mask)

train_X, name_mask, desc_mask, all_mask = feature_extract(train)
fm_model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=train_X.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                    D_fm=200, e_noise=0.0001, iters=17, inv_link="identity", threads=4)
fm_model.fit(train_X, train_y)
print('[{}] Train FM completed'.format(time.time() - start_time))

if develop:
    dev['general_cat'], dev['subcat_1'], dev['subcat_2'] = \
        zip(*dev['category_name'].apply(lambda x: split_cat(x)))
    dev.drop('category_name', axis=1, inplace=True)
    handle_missing_inplace(dev)
    cutting(dev)
    text_length_feature(dev)
    normalize_dataset_text(dev)
    delete_unseen(dev)
    to_categorical(dev)
    
    ## RNN
    dev['encoded_brand_name'] = le_brand.transform(dev['brand_name'])
    dev['encoded_general_cat'] = le_general_cat.transform(dev['general_cat'])
    dev['encoded_subcat_1'] = le_subcat_1.transform(dev['subcat_1'])
    dev['encoded_subcat_2'] = le_subcat_2.transform(dev['subcat_2'])
    
    dev['seq_item_description'] = tok_desc.texts_to_sequences(dev["item_description"].values)
    dev['seq_name'] = tok_name.texts_to_sequences(dev["name"].values)
    
    dev_X = get_rnn_data(dev)
    preds_rnn = model.predict(dev_X)
    print("RNN dev RMSLE:", rmsle(np.expm1(dev_y), np.expm1(preds_rnn.flatten())))
    
    dev.drop(['encoded_brand_name', 'encoded_general_cat','encoded_subcat_1',
              'encoded_subcat_2', 'seq_item_description', 'seq_name'], axis=1, inplace=True)
    del dev_X
    gc.collect()

    ## FM
    dev_X, _, _, _ = feature_extract(dev, train = False, 
                                    name_mask = name_mask, 
                                    desc_mask = desc_mask, 
                                    all_mask = all_mask)
    preds_fm = fm_model.predict(dev_X)
    print("FM_FTRL dev RMSLE:", rmsle(np.expm1(dev_y), np.expm1(preds_fm)))
    
    ## Ensemble
    preds = preds_rnn.flatten()*0.5 + preds_fm * 0.6
    print("FM + RNN Ensemble RMSLE:", rmsle(np.expm1(dev_y), np.expm1(preds)))
else:
    test = pd.read_table('../input/test.tsv', engine='c')
    ## Pre-processing
    test['general_cat'], test['subcat_1'], test['subcat_2'] = \
        zip(*test['category_name'].apply(lambda x: split_cat(x)))
    test.drop('category_name', axis=1, inplace=True)
    handle_missing_inplace(test)
    cutting(test)
    text_length_feature(test)
    normalize_dataset_text(test)
    delete_unseen(test)
    to_categorical(test)
    
    ## RNN
    test['encoded_brand_name'] = le_brand.transform(test['brand_name'])
    test['encoded_general_cat'] = le_general_cat.transform(test['general_cat'])
    test['encoded_subcat_1'] = le_subcat_1.transform(test['subcat_1'])
    test['encoded_subcat_2'] = le_subcat_2.transform(test['subcat_2'])
    
    test['seq_item_description'] = tok_desc.texts_to_sequences(test["item_description"].values)
    test['seq_name'] = tok_name.texts_to_sequences(test["name"].values)
    
    test_X = get_rnn_data(test)
    preds_rnn = model.predict(test_X)
    
    test.drop(['encoded_brand_name', 'encoded_general_cat','encoded_subcat_1',
              'encoded_subcat_2', 'seq_item_description', 'seq_name'], axis=1, inplace=True)
    del test_X
    gc.collect()

    ## FM
    test_X, _, _, _ = feature_extract(test, train = False, 
                                    name_mask = name_mask, 
                                    desc_mask = desc_mask, 
                                    all_mask = all_mask)
    preds_fm = fm_model.predict(test_X)
    ## Ensemble
    preds = preds_rnn.flatten()*0.5 + preds_fm * 0.5
    submission = test[['test_id']]
    submission['price'] = np.expm1(preds)
    submission.to_csv("submission_rnn_fm_normalize.csv", index=False)
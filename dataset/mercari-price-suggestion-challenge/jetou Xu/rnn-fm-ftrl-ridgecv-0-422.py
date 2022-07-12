# Based on Bojan -> https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44944
# and Nishant -> https://www.kaggle.com/nishkgp/more-improved-ridge-2-lgbm

import gc
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from collections import defaultdict
from glob import glob
import sys
import math

#Add https://www.kaggle.com/anttip/wordbatch to your kernel Data Sources, 
#until Kaggle admins fix the wordbatch pip package installation
sys.path.insert(0, '../input/wordbatch/wordbatch/')
import wordbatch

from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL

from nltk.corpus import stopwords
import re

def aggregate_predicts3(Y1, Y2, Y3, ratio1, ratio2):
    assert Y1.shape == Y2.shape
    return Y1 * ratio1 + Y2 * ratio2 + Y3 * (1.0 - ratio1-ratio2)

# Based on Bojan's -> https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44944
# Changes:
# 1. Split category_name into sub-categories
# 2. Parallelize LGBM to 4 cores
# 3. Increase the number of rounds in 1st LGBM
# 4. Another LGBM with different seed for model and training split, slightly different hyper-parametes.
# 5. Weights on ensemble
# 6. SGDRegressor doesn't improve the result, going with only 1 Ridge and 2 LGBM
#remove zero price items
import pyximport; pyximport.install()
import gc
import time
debug = False
from sklearn.decomposition import TruncatedSVD
#svd = TruncatedSVD(n_components=1000, random_state=42)

from joblib import Parallel, delayed

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDRegressor
import lightgbm as lgb

NUM_BRANDS = 4500
NUM_CATEGORIES = 1200

NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 110000
develop = False

    
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation
# from keras.layers import Bidirectional
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from nltk.corpus import stopwords
import math
import pyximport
pyximport.install()
import os
import random
import numpy as np
import tensorflow as tf
os.environ['PYTHONHASHSEED'] = '10000'
np.random.seed(10001)
random.seed(10002)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=5, inter_op_parallelism_threads=1)


def rmsle2(Y, Y_pred):
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))
    
def aggregate_predicts3(Y1, Y2, Y3, ratio1, ratio2):
    assert Y1.shape == Y2.shape
    return Y1 * ratio1 + Y2 * ratio2 + Y3 * (1.0 - ratio1-ratio2)
  
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
        
        
        
        
def aggregate_predicts3(Y1, Y2, Y3, ratio1, ratio2):
    assert Y1.shape == Y2.shape
    return Y1 * ratio1 + Y2 * ratio2 + Y3 * (1.0 - ratio1-ratio2)

def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
    

    
    
def fill_missing_values(df):
    df.category_name.fillna(value="missing", inplace=True)
    df.brand_name.fillna(value="missing", inplace=True)
    df.item_description.fillna(value="missing", inplace=True)
    df.item_description.replace('No description yet',"missing", inplace=True)
    return df
    



def root_mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1)+0.0000001)
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)+0.0000001)


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


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


def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])
import re
def preprocess(string):
    string = string.lower()  # If strings not lowercased
    item_list = {r"16 gb": r"16gb", r"32 gb": "32gb", r"64 gb": r"64gb", r"128 gb": "128gb", 
    r"256 gb": "256gb", r'14 k':'14k', r'iphone6+':'iphone 6 plus', r'500 gb': '500gb',
    r"14 kt": "14kt",}
    # Continue adding more substitutions for normalization
    for i in item_list.keys():
      string = re.sub(i, item_list[i], string)
    return string

def main():
    def brandfinder(line):
        brand = line[0]
        name = line[1]
        namesplit = name.split(' ')
        if brand == 'missing':
            for x in namesplit:
                if x in all_brands:
                    return name
        if name in all_brands:
            return name
        return brand
    def new_rnn_model(lr=0.001, decay=0.0):
    # Inputs
        name = Input(shape=[X_train["name"].shape[1]], name="name")
        item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
        brand_name = Input(shape=[1], name="brand_name")
    #     category = Input(shape=[1], name="category")
    #     category_name = Input(shape=[X_train["category_name"].shape[1]], name="category_name")
        item_condition = Input(shape=[1], name="item_condition")
        num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
        desc_len = Input(shape=[1], name="desc_len")
        name_len = Input(shape=[1], name="name_len")
        subcat_0 = Input(shape=[1], name="subcat_0")
        subcat_1 = Input(shape=[1], name="subcat_1")
        subcat_2 = Input(shape=[1], name="subcat_2")
    
        # Embeddings layers (adjust outputs to help model)
        emb_name = Embedding(MAX_TEXT, 20)(name)
        emb_item_desc = Embedding(MAX_TEXT, 60)(item_desc)
        emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    #     emb_category_name = Embedding(MAX_TEXT, 20)(category_name)
    #     emb_category = Embedding(MAX_CATEGORY, 10)(category)
        emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
        emb_desc_len = Embedding(MAX_DESC_LEN, 5)(desc_len)
        emb_name_len = Embedding(MAX_NAME_LEN, 5)(name_len)
        emb_subcat_0 = Embedding(MAX_SUBCAT_0, 10)(subcat_0)
        emb_subcat_1 = Embedding(MAX_SUBCAT_1, 10)(subcat_1)
        emb_subcat_2 = Embedding(MAX_SUBCAT_2, 10)(subcat_2)
        
    
        # rnn layers (GRUs are faster than LSTMs and speed is important here)
        rnn_layer1 = GRU(16) (emb_item_desc)
        rnn_layer2 = GRU(8) (emb_name)
        rnn_layer3 = GRU(4) (emb_brand_name)
    
        # main layers
        main_l = concatenate([
            # Flatten() (emb_brand_name)
    #         , Flatten() (emb_category)
            Flatten() (emb_item_condition)
            , Flatten() (emb_desc_len)
            , Flatten() (emb_name_len)
            , Flatten() (emb_subcat_1)
            , Flatten() (emb_subcat_2)
            , rnn_layer1
            , rnn_layer2
            , rnn_layer3
            , num_vars
        ])
        # (incressing the nodes or adding layers does not effect the time quite as much as the rnn layers)
        main_l = Dropout(0.1)(Dense(512,kernel_initializer='normal',activation='relu') (main_l))
        main_l = Dropout(0.1)(Dense(256,kernel_initializer='normal',activation='relu') (main_l))
        main_l = Dropout(0.1)(Dense(128,kernel_initializer='normal',activation='relu') (main_l))
        main_l = Dropout(0.1)(Dense(64,kernel_initializer='normal',activation='relu') (main_l))
    
    
        # the output layer.
        output = Dense(1, activation="linear") (main_l)
        
        model = Model([name, item_desc, brand_name , item_condition, 
                       num_vars, desc_len, name_len, subcat_0, subcat_1, subcat_2], output)
    
        optimizer = Adam(lr=lr, decay=decay)
        # (mean squared error loss function works as well as custom functions)  
        model.compile(loss = 'mse', optimizer = optimizer)
    
        return model
        
    def get_rnn_data(dataset):
        X = {
            'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
            'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
            'brand_name': np.array(dataset.brand_name),
            'category': np.array(dataset.category),
    #         'category_name': pad_sequences(dataset.seq_category, maxlen=MAX_CATEGORY_SEQ),
            'item_condition': np.array(dataset.item_condition_id),
            'num_vars': np.array(dataset[["shipping"]]),
            'desc_len': np.array(dataset[["desc_len"]]),
            'name_len': np.array(dataset[["name_len"]]),
            'subcat_0': np.array(dataset.subcat_0),
            'subcat_1': np.array(dataset.subcat_1),
            'subcat_2': np.array(dataset.subcat_2),
        }
        return X

    
    train_df = pd.read_table('../input/mercari-price-suggestion-challenge/train.tsv')
    test_df = pd.read_table('../input/mercari-price-suggestion-challenge/test.tsv')
    print(train_df.shape, test_df.shape)
    
    train_df = train_df.drop(train_df[(train_df.price < 1.0)].index)
    train_df.shape
    submission: pd.DataFrame = test_df[['test_id']]
    
    



    train_df['desc_len'] = train_df['item_description'].apply(lambda x: wordCount(x))
    test_df['desc_len'] = test_df['item_description'].apply(lambda x: wordCount(x))
    train_df['name_len'] = train_df['name'].apply(lambda x: wordCount(x))
    test_df['name_len'] = test_df['name'].apply(lambda x: wordCount(x))


    
    
    
    train_df['subcat_0'], train_df['subcat_1'], train_df['subcat_2'] = \
    zip(*train_df['category_name'].apply(lambda x: split_cat(x)))
    test_df['subcat_0'], test_df['subcat_1'], test_df['subcat_2'] = \
    zip(*test_df['category_name'].apply(lambda x: split_cat(x)))
    
    
    full_set = pd.concat([train_df,test_df])
    all_brands = set(full_set['brand_name'].values)
    train_df.brand_name.fillna(value="missing", inplace=True)
    test_df.brand_name.fillna(value="missing", inplace=True)
    
    # get to finding!
    premissing = len(train_df.loc[train_df['brand_name'] == 'missing'])

    train_df['brand_name'] = train_df[['brand_name','name']].apply(brandfinder, axis = 1)
    test_df['brand_name'] = test_df[['brand_name','name']].apply(brandfinder, axis = 1)
    found = premissing-len(train_df.loc[train_df['brand_name'] == 'missing'])
    
    train_df["target"] = np.log1p(train_df.price)
    
    # Split training examples into train/dev examples.
    if debug:
        train_df, dev_df = train_test_split(train_df, random_state=123, train_size=0.99)
        n_devs = dev_df.shape[0]
        print("Validating on", n_devs, "examples")
    # Calculate number of train/dev/test examples.
    n_trains = train_df.shape[0]
    n_tests = test_df.shape[0]
    print("Training on", n_trains, "examples")
    print("Testing on", n_tests, "examples")
    
    if debug:
        full_df = pd.concat([train_df, dev_df, test_df])
        
    full_df = pd.concat([train_df,test_df])
    

    
    print("Filling missing data...")
    full_df = fill_missing_values(full_df)
    print(full_df.category_name[1])
    
    
    print("Processing categorical data...")
    le = LabelEncoder()
    # full_df.category = full_df.category_name
    le.fit(full_df.category_name)
    full_df['category'] = le.transform(full_df.category_name)
    
    le.fit(full_df.brand_name)
    full_df.brand_name = le.transform(full_df.brand_name)
    
    le.fit(full_df.subcat_0)
    full_df.subcat_0 = le.transform(full_df.subcat_0)
    
    le.fit(full_df.subcat_1)
    full_df.subcat_1 = le.transform(full_df.subcat_1)
    
    le.fit(full_df.subcat_2)
    full_df.subcat_2 = le.transform(full_df.subcat_2)
    
    del le
    
    
    print("Transforming text data to sequences...")
    raw_text = np.hstack([full_df.item_description.str.lower(), full_df.name.str.lower(), full_df.category_name.str.lower()])
    
    print("   Fitting tokenizer...")
    tok_raw = Tokenizer()
    tok_raw.fit_on_texts(raw_text)
    
    print("   Transforming text to sequences...")
    full_df['seq_item_description'] = tok_raw.texts_to_sequences(full_df.item_description.str.lower())
    full_df['seq_name'] = tok_raw.texts_to_sequences(full_df.name.str.lower())
    # full_df['seq_category'] = tok_raw.texts_to_sequences(full_df.category_name.str.lower())
    
    del tok_raw
    
    
    
    
    MAX_NAME_SEQ = 10 #17
    MAX_ITEM_DESC_SEQ = 75 #269
    MAX_CATEGORY_SEQ = 8 #8
    MAX_TEXT = np.max([
        np.max(full_df.seq_name.max()),
        np.max(full_df.seq_item_description.max()),
    #     np.max(full_df.seq_category.max()),
    ]) + 100
    MAX_CATEGORY = np.max(full_df.category.max()) + 1
    MAX_BRAND = np.max(full_df.brand_name.max()) + 1
    MAX_CONDITION = np.max(full_df.item_condition_id.max()) + 1
    MAX_DESC_LEN = np.max(full_df.desc_len.max()) + 1
    MAX_NAME_LEN = np.max(full_df.name_len.max()) + 1
    MAX_SUBCAT_0 = np.max(full_df.subcat_0.max()) + 1
    MAX_SUBCAT_1 = np.max(full_df.subcat_1.max()) + 1
    MAX_SUBCAT_2 = np.max(full_df.subcat_2.max()) + 1
    



    train = full_df[:n_trains]
    test = full_df[n_trains:]
    if debug:
        dev = full_df[n_trains:n_trains+n_devs]
        test = full_df[n_trains+n_devs:]
    
    X_train = get_rnn_data(train)
    Y_train = train.target.values.reshape(-1, 1)
    if debug:
        X_dev = get_rnn_data(dev)
        Y_dev = dev.target.values.reshape(-1, 1)
    
    X_test = get_rnn_data(test)
    del full_df

    
    
    
    np.random.seed(123)



    model = new_rnn_model()
    model.summary()
    del model
    
    BATCH_SIZE = 620*2
    epochs = 1
    
    # Calculate learning rate decay.
    exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
    steps = int(len(X_train['name']) / BATCH_SIZE) * epochs
    lr_init, lr_fin = 0.005, 0.001
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    
    # Create model and fit it with training dataset.
    rnn_model = new_rnn_model(lr=lr_init, decay=lr_decay)
    if debug:
        rnn_model.fit(
                X_train, Y_train, epochs=epochs, batch_size=BATCH_SIZE,
                validation_data=(X_dev, Y_dev), verbose=1,
        )
    rnn_model.fit(
        X_train, Y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=1,
    )
    
    
    print("Evaluating the model on validation data...")
    if debug:
        Y_dev_preds_rnn = rnn_model.predict(X_dev, batch_size=BATCH_SIZE)
        print(" RMSLE error:", rmsle2(Y_dev, Y_dev_preds_rnn))
    
    rnn_preds = rnn_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    rnn_preds = np.expm1(rnn_preds)
    
    
    del rnn_model
    del train
    del test
    del X_train
    del Y_train
    del X_test
    del train_df
    del test_df
    
    gc.collect()
    
    
    start_time = time.time()
    from time import gmtime, strftime
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    # # if 1 == 1:
    train = pd.read_table('../input/mercari-price-suggestion-challenge/train.tsv', engine='c')
    test = pd.read_table('../input/mercari-price-suggestion-challenge/test.tsv', engine='c')

    # train = pd.read_table('../input/train.tsv', engine='c')
    # test = pd.read_table('../input/test.tsv', engine='c')

    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)
    nrow_test = train.shape[0]  # -dftt.shape[0]
    dftt = train[(train.price < 1.0)]
    train = train.drop(train[(train.price < 1.0)].index)
    del dftt['price']
    nrow_train = train.shape[0]
    # print(nrow_train, nrow_test)
    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, dftt, test])

    # submission2: pd.DataFrame = test[['test_id']]
    # merge = full_set
    # del full_set



    del train
    del test
    gc.collect()

    merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
        zip(*merge['category_name'].apply(lambda x: split_cat(x)))
    merge.drop('category_name', axis=1, inplace=True)
    print('[{}] Split categories completed.'.format(time.time() - start_time))

    handle_missing_inplace(merge)
    print('[{}] Handle missing completed.'.format(time.time() - start_time))

    cutting(merge)
    print('[{}] Cut completed.'.format(time.time() - start_time))

    to_categorical(merge)
    print('[{}] Convert categorical completed'.format(time.time() - start_time))
    merge['item_description'] = merge.item_description.apply(preprocess)
    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                                  "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                                  "idf": None,
                                                                  }), procs=8)
    wb.dictionary_freeze= True
    X_name2 = wb.fit_transform(merge['name'])
    del(wb)
    X_name2 = X_name2[:, np.array(np.clip(X_name2.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))

    # cv = CountVectorizer(min_df=NAME_MIN_DF,ngram_range=(1, 2),
    #     token_pattern=r'\b\w+\b|\w?-\w+', stop_words = 'english')
    # X_name = cv.fit_transform(merge['name'])
    print('[{}] Count vectorize `name` completed.'.format(time.time() - start_time))

    cv = CountVectorizer()
    X_category1 = cv.fit_transform(merge['general_cat'])
    X_category2 = cv.fit_transform(merge['subcat_1'])
    X_category3 = cv.fit_transform(merge['subcat_2'])
    del(cv)
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

    # tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
    #                      ngram_range=(1, 2),
    #                      token_pattern=r'\w+|\d\w+',)
    # X_description = tv.fit_transform(merge['item_description'])
    # del(tv)
    print('[{}] TFIDF vectorize `item_description` completed.'.format(time.time() - start_time))
    
        # wb= wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 3, "hash_ngrams_weights": [1.0, 1.0, 0.5],
    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                                  "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                                  "idf": None})
                             , procs=8)
    wb.dictionary_freeze= True
    X_description2 = wb.fit_transform(merge['item_description'])
    del(wb)
    X_description2 = X_description2[:, np.array(np.clip(X_description2.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    print('[{}] Vectorize `item_description2` completed.'.format(time.time() - start_time))

    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name'])
    print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))
    del(lb)
    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
    print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))
    # print (X_dummies.shape, X_description.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape, X_name.shape)
    sparse_merge2 = hstack((X_dummies, X_description2, X_brand, X_category1, X_category2, X_category3, X_name2)).tocsr()
    print('[{}] Create sparse merge completed'.format(time.time() - start_time))
    
    
    
    mask = np.array(np.clip(sparse_merge2.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    sparse_merge2 = sparse_merge2[:, mask]
    X = sparse_merge2[:nrow_train]
    X_test = sparse_merge2[nrow_test:]
    print(sparse_merge2.shape)

    gc.collect()
    train_X, train_y = X, y
    # if develop:
    #     train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.05, random_state=100)

    # model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=sparse_merge2.shape[1], iters=50, inv_link="identity", threads=1)
    model = RidgeCV(
    fit_intercept=True, alphas=[5.0],
    normalize=False, cv = 2, scoring='neg_mean_squared_error',
    )
    model.fit(train_X, train_y)
    print('[{}] Train FTRL completed'.format(time.time() - start_time))
    if develop:
        preds = model.predict(X=valid_X)
        print("FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))

    predsF = model.predict(X_test)
    predsF = np.expm1(predsF)
    predsF = predsF.reshape(-1, 1)

    print('[{}] Predict FTRL completed'.format(time.time() - start_time))

    model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=sparse_merge2.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                    D_fm=200, e_noise=0.0001, iters=15, inv_link="identity", threads=4)

    model.fit(train_X, train_y)
    print('[{}] Train ridge v2 completed'.format(time.time() - start_time))
    if develop:
        preds = model.predict(X=valid_X)
        print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))

    predsFM = model.predict(X_test)
    predsFM = np.expm1(predsFM)
    predsFM = predsFM.reshape(-1, 1)

    print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))
    
    del(sparse_merge2)
    del(train_X)
    del(train_y)
    del(model)
    del(X)
    del(X_test)
    del(X_description2)
    del(X_name2)
    del(merge)
    gc.collect()
  
    
    

    # train_X2, valid_X2, train_y2, valid_y2 = train_test_split(X, y, test_size = 0.1, random_state = 101) 
    # d_train2 = lgb.Dataset(train_X2, label=train_y2, max_bin=8192)
    # d_valid2 = lgb.Dataset(valid_X2, label=valid_y2, max_bin=8192)
    # watchlist2 = [d_train2, d_valid2]

    # model = lgb.train(params2, train_set=d_train2, num_boost_round=5000, valid_sets=watchlist2, \
    # early_stopping_rounds=1000, verbose_eval=1000) 
    # predsL2 = model.predict(X_test)

    # print('[{}] Predict lgb 2 completed.'.format(time.time() - start_time))
    preds = aggregate_predicts3(predsFM, predsF, rnn_preds, 0.3, 0.2)
    kk = pd.DataFrame({
            "test_id": submission.test_id,
            "price": preds.reshape(-1),
    })
    kk.to_csv("./rnn_ridge_submission_var_1.csv", index=False)

    preds = aggregate_predicts3(predsFM, predsF, rnn_preds, 0.5, 0.2)
    kk = pd.DataFrame({
            "test_id": submission.test_id,
            "price": preds.reshape(-1),
    })
    kk.to_csv("./rnn_ridge_submission_var_2.csv", index=False)
    
    preds = aggregate_predicts3(predsFM, predsF, rnn_preds, 0.4, 0.1)
    kk = pd.DataFrame({
            "test_id": submission.test_id,
            "price": preds.reshape(-1),
    })
    kk.to_csv("./rnn_ridge_submission_var_3.csv", index=False)
    
    preds = aggregate_predicts3(predsFM, predsF, rnn_preds, 0.4, 0.2)
    kk = pd.DataFrame({
            "test_id": submission.test_id,
            "price": preds.reshape(-1),
    })
    kk.to_csv("./rnn_ridge_submission_var_5.csv", index=False)



if __name__ == '__main__':
    main()
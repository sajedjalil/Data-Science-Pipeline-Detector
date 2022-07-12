# -*- coding: utf-8 -*-
# @Time    : 1/25/18 5:03 PM
# @Author  : LeeYun
# @File    : model_ensemble.py
'''Description :
'''
import os, string, pickle, re, time, gc, multiprocessing, wordbatch, math
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix, hstack
from fastcache import clru_cache as lru_cache
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from keras import backend, optimizers, callbacks
from keras.models import Model
from keras.backend import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Lambda
from wordbatch.models import FM_FTRL
from wordbatch.extractors import WordBag

DEVELOP = False
EPOCH = 3

if DEVELOP:
    TRAIN_SIZE = 10000
else:
    TRAIN_SIZE = 1481661
PRICE_MEAN = 2.98081628517
PRICE_STD = 0.7459273872548303
NumWords = 50000
THREAD = 4
MISSVALUE = 'missvalue'

# 0.416 ensemble
param_space_best_vanila_con1d = {
    'denselayer_units': 244,
    'description_Len': 85,
    'name_Len': 10,
    'embed_name': 58,
    'embed_desc': 52,
    'embed_brand': 22,
    'embed_cat_2': 14,
    'embed_cat_3': 37,
    'name_filter': 114,
    'desc_filter': 96,
    'name_filter_size': 4,
    'desc_filter_size': 4,
    'lr': 0.003924230325700921,
    'batch_size': 933,
    'dense_drop': 0.009082372998548981,
}

# 0.41422
param_space_best_vanila_embend = { 
    'denselayer_units': 244,
    'description_Len': 85,
    'name_Len': 10,
    'embed_name': 248,
    'embed_desc': 223,
    'embed_brand': 22,
    'embed_cat_2': 14,
    'embed_cat_3': 37,
    'lr': 0.004565707978962733,
    'batch_size': 1177,
    'dense_drop': 0.009082372998548981,
}

# 0.41217
param_space_best_FM_FTRL = {
    'alpha': 0.03273793453882604,
    'beta': 0.0011705530094547533,
    'L1': 3.59507400149913e-05,
    'L2': 0.018493058691917252,
    'alpha_fm': 0.015903973928100217,
    'init_fm': 0.02883106077640207,
    'D_fm': 247,
    'e_noise': 0.0003029146164926251,
    'iters': 8,
}

param_space_best_WordBatch = {
    'desc_w1': 1.3740067995315037,
    'desc_w2': 1.0248685266832964,
    'desc_w3': 0.7,
    'name_w1': 2.1385527373939834,
    'name_w2': 0.3894761681383836,
}

param_space_best_ensemble = {
    'embend_weight': 0.6,
    'conv1d_weight': 0.7,
    'FM_FTRL_weight': 1.15,
}
ensemble_total_weights = param_space_best_ensemble['embend_weight'] + param_space_best_ensemble['conv1d_weight'] + \
                         param_space_best_ensemble['FM_FTRL_weight']
param_space_best_ensemble['embend_weight'] /= ensemble_total_weights
param_space_best_ensemble['conv1d_weight'] /= ensemble_total_weights
param_space_best_ensemble['FM_FTRL_weight'] /= ensemble_total_weights
print(param_space_best_ensemble)


@lru_cache(1024)
def split_cat(text: str):
    text = text.split("/")
    if len(text) >= 2:
        return text[0], text[1]
    else:
        return text[0], MISSVALUE


@lru_cache(1024)
def len_splt(str: str):
    return len(str.split(' '))


def prepare_batches(seq: np.ndarray, step: int):
    n = len(seq)
    res = []
    for i in range(0, n, step):
        res.append(seq[i:i + step])
    return res


def text_processor1(text: pd.Series):
    return text.str.lower(). \
        str.replace(r'([a-z]+|[0-9]+)', r' \1 '). \
        str.replace(r'([0-9])( *)\.( *)([0-9])', r'\1.\4'). \
        str.replace('\s+', ' '). \
        str.strip()


def text_processor2(text: pd.Series):
    return text.str.lower(). \
        str.replace(r'[^\.!?@#&%$/\\0-9a-z]', ' '). \
        str.replace(r'(\.|!|\?|@|#|&|%|\$|/|\\|[0-9]+)', r' \1 '). \
        str.replace(r'([0-9])( *)\.( *)([0-9])', r'\1.\4'). \
        str.replace('\s+', ' '). \
        str.strip()


def intersect_cnt(dfs: np.ndarray):
    intersect = np.empty(dfs.shape[0])
    for i in range(dfs.shape[0]):
        obs_tokens = set(dfs[i, 0].split(" "))
        target_tokens = set(dfs[i, 1].split(" "))
        intersect[i] = len(obs_tokens.intersection(target_tokens)) / (obs_tokens.__len__() + 1)
    return intersect


# import random
# def random_words(wc):
#     wc = int(wc)
#     ret = []
#     for _ in range(wc):
#         ret.append(''.join(random.sample(string.ascii_letters, random.randrange(3, 15))))
#     return ret
#
#
# def shuffle_strings(x):
#     global salts
#     x = repr(x).split(' ')
#     i = np.random.randint(len(x))
#     j = np.random.randint(100000)
#     x[i] = salts[j]
#     return ' '.join(x)
#
#
# def shuffle_series(data):
#     global salts
#     for name in ['name', 'item_description']:
#         data[name] = data[name].apply(shuffle_strings)
#     data.loc[0, 'category_name'] = 'a'
#     data.loc[1, 'category_name'] = 'a/b'
#     data.loc[2, 'category_name'] = 'a/b/c'
#     data.loc[3, 'category_name'] = 'asd/asediafi/as'
#     data.loc[4, 'category_name'] = 'scas/asdc/asdc'
#     data.loc[5, 'category_name'] = 'Men/shirt'
#     data.loc[6, 'category_name'] = 'Men'
#     data.loc[7, 'category_name'] = 'Men/shirt/shoe'
#     data.loc[8, 'category_name'] = 'a'
#     data.loc[10, 'brand_name'] = 'a'
#     data.loc[11, 'brand_name'] = 'a b'
#     data.loc[12, 'brand_name'] = 'a/b/c'
#     data.loc[13, 'brand_name'] = 'asdasediafi 88as'
#     data.loc[14, 'brand_name'] = 'scas*asdc & asdc'
#     data.loc[15, 'brand_name'] = 'Men - shirt'
#     data.loc[16, 'brand_name'] = 'Men'
#     data.loc[17, 'brand_name'] = 'Men3shirtshoe'
#     data.loc[18, 'brand_name'] = 'a'
#     return data

def get_extract_feature():
    def read_file(name: str):
        source = '../input/%s.tsv' % name
        df = pd.read_table(source, engine='c')
        return df

    def textclean(merge: pd.DataFrame):
        columns = ['name', 'category_name','brand_name', 'item_description']
        for col in columns: merge[col].fillna(value=MISSVALUE, inplace=True)
        merge['item_condition_id'].fillna(value=1, inplace=True)
        merge['item_condition_id'] = merge['item_condition_id'].astype('int32')
        merge['shipping'].fillna(value=0, inplace=True)
        merge['shipping'] = merge['shipping'].astype('int32')

        start_time = time.time()

        columns = ['item_description', 'name']
        p = multiprocessing.Pool(THREAD)
        length = merge.shape[0]
        len1, len2, len3 = length // 4, length // 2, (length // 4) * 3
        for col in columns:
            print(col)
            slices = [merge[col][:len1], merge[col][len1:len2], merge[col][len2:len3], merge[col][len3:]]
            dfvalue = []
            dfs = p.imap(text_processor1, slices)
            for df in dfs: dfvalue.append(df.values)
            merge[col] = np.concatenate((dfvalue[0], dfvalue[1], dfvalue[2], dfvalue[3]))
        p.close();
        slices, dfvalue, dfs, df, p = None, None, None, None, None;
        gc.collect()

        print('[{}] clean item_description completed'.format(time.time() - start_time))

        columns = ['brand_name', 'category_name']
        p = multiprocessing.Pool(THREAD)
        length = merge.shape[0]
        len1, len2, len3 = length // 4, length // 2, (length // 4) * 3
        for col in columns:
            slices = [merge[col][:len1], merge[col][len1:len2], merge[col][len2:len3], merge[col][len3:]]
            dfvalue = []
            dfs = p.imap(text_processor2, slices)
            for df in dfs: dfvalue.append(df.values)
            merge[col] = np.concatenate((dfvalue[0], dfvalue[1], dfvalue[2], dfvalue[3]))
        p.close();
        slices, dfvalue, dfs, df, p = None, None, None, None, None;
        gc.collect()

        merge['category_1'], merge['category_2'] = zip(*merge['category_name'].apply(lambda x: split_cat(x)))
        return merge

    def get_cleaned_data():
        dftrain = read_file('train')
        dftest = read_file('test')

        # global salts
        # # Make a stage 2 test by copying test five times...
        # test1 = dftest.copy()
        # test2 = dftest.copy()
        # test3 = dftest.copy()
        # test4 = dftest.copy()
        # test5 = dftest.copy()
        # dftest = pd.concat([test1, test2, test3, test4, test5, test5[:100000]], axis=0)
        # test1 = None
        # test2 = None
        # test3 = None
        # test4 = None
        # test5 = None
        # dftest['test_id'] = np.arange(dftest.shape[0], dtype=np.int64)
        #
        # word_counts = 100000  # Word Numbers in Stage I denpends on the tokenizer
        # salts = random_words(word_counts)
        # dftest = shuffle_series(dftest)
        # print('shuffle_series done')
        # del salts

        dftrain = dftrain[dftrain.price != 0]
        dfAll = pd.concat((dftrain, dftest), ignore_index=True)
        if DEVELOP: dfAll = dfAll[:20000]
        dfAll = textclean(dfAll)
        submission: pd.DataFrame = dftest[['test_id']]
        dftrain, dftest = None, None;
        gc.collect()
        return dfAll, submission

    def add_Frec_feat(dfAll: pd.DataFrame, col: str):
        s = dfAll[col].value_counts()
        s[MISSVALUE] = 0
        dfAll = dfAll.merge(s.to_frame(name=col + '_Frec'), left_on=col, right_index=True, how='left')
        s = None
        return dfAll

    dfAll, submission = get_cleaned_data()
    print('data cleaned')

    # add the Frec features
    columns = ['brand_name']
    print('add the Frec features')
    for col in columns: dfAll = add_Frec_feat(dfAll, col)

    # intersection count between 'item_description','name','brand_name','category_name'
    columns = [['brand_name', 'name'], ['brand_name', 'item_description']]
    p = multiprocessing.Pool(THREAD)
    length = dfAll.shape[0]
    len1, len2, len3 = length // 4, length // 2, (length // 4) * 3
    for col in columns:
        slices = [dfAll[col].values[:len1], dfAll[col].values[len1:len2], dfAll[col].values[len2:len3],
                  dfAll[col].values[len3:]]
        dfvalue = []
        dfs = p.imap(intersect_cnt, slices)
        for df in dfs: dfvalue.append(df)
        dfAll['%s_%s_Intsct' % (col[0], col[1])] = np.concatenate((dfvalue[0], dfvalue[1], dfvalue[2], dfvalue[3]))
    p.close();
    slices, dfvalue, dfs, df, p = None, None, None, None, None;
    gc.collect()

    # remove brand_name that only appeared in dfTest
    dftrain = dfAll[:TRAIN_SIZE]
    columns = ['category_1', 'category_2', 'category_name', 'brand_name']
    for col in columns:
        mask = ~dfAll[col].isin(dftrain[col].unique())
        dfAll.loc[mask, col] = MISSVALUE
        print('%s missvalue %d' % (col, mask.sum()))

    # count item_description length
    dfAll['item_description_wordLen'] = dfAll.item_description.apply(lambda x: len_splt(x))
    # nomalize price
    y_train = ((np.log1p(dfAll.price) - PRICE_MEAN) / PRICE_STD).values[:TRAIN_SIZE].reshape(-1, 1).astype(np.float32)
    dfAll.drop(['price', 'test_id', 'train_id'], axis=1, inplace=True)
    return dfAll, submission, y_train


def Label_Encoder(df: pd.Series):
    le = LabelEncoder()
    return le.fit_transform(df)


def Item_Tokenizer(df: pd.Series):
    # do not concern the words only appeared in dftest
    tok_raw = Tokenizer(num_words=NumWords, filters='')
    tok_raw.fit_on_texts(df[:TRAIN_SIZE])
    return tok_raw.texts_to_sequences(df), min(tok_raw.word_counts.__len__() + 1, NumWords)


def Preprocess_features(merge: pd.DataFrame, start_time):
    merge['item_condition_id'] = merge['item_condition_id'].astype('category')
    print('[{}] Convert categorical completed'.format(time.time() - start_time))

    Item_size = {}
    # Label_Encoder brand_name + category
    columns = ['category_1', 'category_2', 'category_name', 'brand_name']
    p = multiprocessing.Pool(THREAD)
    dfs = p.imap(Label_Encoder, [merge[col] for col in columns])
    for col, df in zip(columns, dfs):
        merge['Lb_' + col] = df
        Item_size[col] = merge['Lb_' + col].max() + 1
    print('[{}] Label Encode `brand_name` and `categories` completed.'.format(time.time() - start_time))
    p.close();
    dfs, df, p = None, None, None;
    gc.collect()

    # sequance item_description,name
    columns = ['item_description', 'name']
    p = multiprocessing.Pool(THREAD)
    dfs = p.imap(Item_Tokenizer, [merge[col] for col in columns])
    for col, df in zip(columns, dfs):
        merge['Seq_' + col], Item_size[col] = df
    print('[{}] sequance `item_description` and `name` completed.'.format(time.time() - start_time))
    print(Item_size)
    p.close();
    dfs, df, p = None, None, None;
    gc.collect()

    # hand feature
    columns = ['brand_name_Frec', 'item_description_wordLen']
    for col in columns:
        merge[col] = np.log1p(merge[col])
        merge[col] = merge[col] / merge[col].max()

    # reduce memory
    for col in merge.columns:
        if str(merge[col].dtype) == 'int64':
            merge[col] = merge[col].astype('int32')
        elif str(merge[col].dtype) == 'float64':
            merge[col] = merge[col].astype('float32')

    hand_feature = ['brand_name_Frec',
                    'item_description_wordLen',
                    'brand_name_name_Intsct',
                    'brand_name_item_description_Intsct']
    return merge, Item_size, hand_feature


def embed(inputs, size, dim):
    std = np.sqrt(2 / dim)
    emb = tf.Variable(tf.random_uniform([size, dim], -std, std))
    lookup = tf.nn.embedding_lookup(emb, inputs)
    return lookup


def conv1d(inputs, num_filters, filter_size, padding='same', strides=1):
    he_std = np.sqrt(2 / (filter_size * num_filters))
    out = tf.layers.conv1d(
        inputs=inputs, filters=num_filters, padding=padding,
        kernel_size=filter_size,
        strides=strides,
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(stddev=he_std))
    return out


def dense(X, size, activation=None):
    he_std = np.sqrt(2 / int(X.shape[1]))
    out = tf.layers.dense(X, units=size, activation=activation,
                          kernel_initializer=tf.random_normal_initializer(stddev=he_std))
    return out


def CountVector(df):
    wb = CountVectorizer()
    return wb.fit_transform(df).astype(np.int32)


def Get_CountVectorizer(merge: pd.DataFrame):
    columns = ['category_2', 'category_name', 'brand_name']
    p = multiprocessing.Pool(THREAD)
    dfs = p.imap(CountVector, [merge[col] for col in columns])
    results = []
    for col, df in zip(columns, dfs): results.append(df)
    p.close();
    dfs, df, p = None, None, None;
    gc.collect()
    return results[0], results[1], results[2]


def LabelBinarize(df):
    lb = LabelBinarizer(sparse_output=True)
    return lb.fit_transform(df).astype(np.int32)


def Get_LabelBinarizer(merge: pd.DataFrame):
    columns = ['category_1', 'category_name', 'brand_name', 'gencat_cond', 'subcat_1_cond', 'subcat_2_cond']
    p = multiprocessing.Pool(THREAD)
    dfs = p.imap(LabelBinarize, [merge[col] for col in columns])
    results = []
    for col, df in zip(columns, dfs): results.append(df)
    p.close();
    dfs, df, p = None, None, None;
    gc.collect()
    return results[0], results[1], results[2], results[3], results[4], results[5]


def Split_Train_Test_NN(data: pd.DataFrame, hand_feature):
    param_dict = param_space_best_vanila_con1d
    X_seq_item_description = pad_sequences(data['Seq_item_description'], maxlen=param_dict['description_Len'])
    X_seq_name = pad_sequences(data['Seq_name'], maxlen=param_dict['name_Len'])

    X_brand_name = data.Lb_brand_name.values.reshape(-1, 1)
    X_category_1 = data.Lb_category_1.values.reshape(-1, 1)
    X_category_2 = data.Lb_category_2.values.reshape(-1, 1)
    X_category_name = data.Lb_category_name.values.reshape(-1, 1)
    X_item_condition_id = (data.item_condition_id.values.astype(np.int32) - 1).reshape(-1, 1)
    X_shipping = ((data.shipping.values - data.shipping.values.mean()) / data.shipping.values.std()).reshape(-1, 1)
    X_hand_feature = (data[hand_feature].values - data[hand_feature].values.mean(axis=0)) / data[
        hand_feature].values.std(axis=0)

    X_train = dict(
        X_seq_item_description=X_seq_item_description[:TRAIN_SIZE],
        X_seq_name=X_seq_name[:TRAIN_SIZE],
        X_brand_name=X_brand_name[:TRAIN_SIZE],
        X_category_1=X_category_1[:TRAIN_SIZE],
        X_category_2=X_category_2[:TRAIN_SIZE],
        X_category_name=X_category_name[:TRAIN_SIZE],
        X_item_condition_id=X_item_condition_id[:TRAIN_SIZE],
        X_shipping=X_shipping[:TRAIN_SIZE],
        X_hand_feature=X_hand_feature[:TRAIN_SIZE],
    )
    X_valid = dict(
        X_seq_item_description=X_seq_item_description[TRAIN_SIZE:],
        X_seq_name=X_seq_name[TRAIN_SIZE:],
        X_brand_name=X_brand_name[TRAIN_SIZE:],
        X_category_1=X_category_1[TRAIN_SIZE:],
        X_category_2=X_category_2[TRAIN_SIZE:],
        X_category_name=X_category_name[TRAIN_SIZE:],
        X_item_condition_id=X_item_condition_id[TRAIN_SIZE:],
        X_shipping=X_shipping[TRAIN_SIZE:],
        X_hand_feature=X_hand_feature[TRAIN_SIZE:],
    )
    X_seq_item_description, X_seq_name, X_brand_name, X_category_1, X_category_2, X_category_name, \
    X_item_condition_id, X_shipping, X_hand_feature = None, None, None, None, None, None, None, None, None
    return X_train, X_valid


def Split_Train_Test_FTRL(merge: pd.DataFrame, hand_feature, start_time):
    desc_w1 = param_space_best_WordBatch['desc_w1']
    desc_w2 = param_space_best_WordBatch['desc_w2']
    desc_w3 = param_space_best_WordBatch['desc_w3']
    name_w1 = param_space_best_WordBatch['name_w1']
    name_w2 = param_space_best_WordBatch['name_w2']

    merge['item_description'] = merge['name'].map(str) + ' . . ' + \
                                merge['item_description'].map(str)

    wb = wordbatch.WordBatch(normalize_text=None, extractor=(WordBag, {
        "hash_ngrams": 2,
        "hash_ngrams_weights": [name_w1, name_w2],
        "hash_size": 2 ** 28,
        "norm": None,
        "tf": 'binary',
        "idf": None,
    }), procs=8)
    wb.dictionary_freeze = True
    X_name = wb.fit_transform(merge['name']).astype(np.float32)
    del wb
    merge.drop(['name'], axis=1, inplace=True)
    X_name = X_name[:, np.array(np.clip(X_name[:TRAIN_SIZE].getnnz(axis=0) - 2, 0, 1), dtype=bool)]
    print(X_name.shape)
    print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))

    wb = wordbatch.WordBatch(normalize_text=None, extractor=(WordBag, {
        "hash_ngrams": 3,
        "hash_ngrams_weights": [desc_w1, desc_w2, desc_w3],
        "hash_size": 2 ** 28,
        "norm": "l2",
        "tf": 1.0,
        "idf": None
    }), procs=8)
    wb.dictionary_freeze = True
    X_description_train = wb.fit_transform(merge['item_description'][:TRAIN_SIZE]).astype(np.float32)
    mask = np.array(np.clip(X_description_train.getnnz(axis=0) - 6, 0, 1), dtype=bool)
    X_description_train = X_description_train[:, mask]
    print('X_description_train done')
    valid_len = merge.shape[0] - TRAIN_SIZE
    valid_len1, valid_len2 = int(valid_len / 3), int(valid_len * 2 / 3)
    X_description_test1 = wb.fit_transform(merge['item_description'][TRAIN_SIZE:TRAIN_SIZE + valid_len1]).astype(
        np.float32)
    X_description_test1 = X_description_test1[:, mask]
    print('X_description_test1 done')
    X_description_test2 = wb.fit_transform(
        merge['item_description'][TRAIN_SIZE + valid_len1:TRAIN_SIZE + valid_len2]).astype(np.float32)
    X_description_test2 = X_description_test2[:, mask]
    print('X_description_test2 done')
    X_description_test3 = wb.fit_transform(merge['item_description'][TRAIN_SIZE + valid_len2:]).astype(np.float32)
    X_description_test3 = X_description_test3[:, mask]
    print('X_description_test3 done')
    del wb, mask
    merge.drop(['item_description'], axis=1, inplace=True)
    print(X_description_train.shape, X_description_test1.shape, X_description_test2.shape, X_description_test3.shape)
    print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))

    merge['gencat_cond'] = merge['category_1'].map(str) + '_' + merge['item_condition_id'].astype(str)
    merge['subcat_1_cond'] = merge['category_2'].map(str) + '_' + merge['item_condition_id'].astype(str)
    merge['subcat_2_cond'] = merge['category_name'].map(str) + '_' + merge['item_condition_id'].astype(str)
    print(f'[{time.time() - start_time}] Categories and item_condition_id concancenated.')

    X_category2, X_category3, X_brand2 = Get_CountVectorizer(merge)
    X_category1, X_category4, X_brand, X_gencat_cond, X_subcat_1_cond, X_subcat_2_cond = Get_LabelBinarizer(merge)
    merge.drop(
        ['category_1', 'category_2', 'category_name', 'brand_name', 'gencat_cond', 'subcat_1_cond', 'subcat_2_cond'],
        axis=1, inplace=True)
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

    X_dummies = csr_matrix(
        pd.get_dummies(merge[['item_condition_id', 'shipping']], sparse=True).values.astype(np.float32))
    merge.drop(['item_condition_id', 'shipping'], axis=1, inplace=True)
    X_hand_feature = merge[hand_feature].values.astype(np.float32)
    merge.drop(hand_feature, axis=1, inplace=True)
    print('-' * 50)

    # coo_matrix
    X_train = hstack((X_dummies[:TRAIN_SIZE],
                      X_brand[:TRAIN_SIZE],
                      X_brand2[:TRAIN_SIZE],
                      X_category1[:TRAIN_SIZE],
                      X_category2[:TRAIN_SIZE],
                      X_category3[:TRAIN_SIZE],
                      X_category4[:TRAIN_SIZE],
                      X_hand_feature[:TRAIN_SIZE],
                      X_name[:TRAIN_SIZE],
                      X_description_train,
                      X_gencat_cond[:TRAIN_SIZE],
                      X_subcat_1_cond[:TRAIN_SIZE],
                      X_subcat_2_cond[:TRAIN_SIZE],
                      ), dtype=np.float32)
    print(X_description_train.shape)
    X_description_train = None
    gc.collect()
    print('-' * 50)
    X_test1 = hstack((X_dummies[TRAIN_SIZE:TRAIN_SIZE + valid_len1],
                      X_brand[TRAIN_SIZE:TRAIN_SIZE + valid_len1],
                      X_brand2[TRAIN_SIZE:TRAIN_SIZE + valid_len1],
                      X_category1[TRAIN_SIZE:TRAIN_SIZE + valid_len1],
                      X_category2[TRAIN_SIZE:TRAIN_SIZE + valid_len1],
                      X_category3[TRAIN_SIZE:TRAIN_SIZE + valid_len1],
                      X_category4[TRAIN_SIZE:TRAIN_SIZE + valid_len1],
                      X_hand_feature[TRAIN_SIZE:TRAIN_SIZE + valid_len1],
                      X_name[TRAIN_SIZE:TRAIN_SIZE + valid_len1],
                      X_description_test1,
                      X_gencat_cond[TRAIN_SIZE:TRAIN_SIZE + valid_len1],
                      X_subcat_1_cond[TRAIN_SIZE:TRAIN_SIZE + valid_len1],
                      X_subcat_2_cond[TRAIN_SIZE:TRAIN_SIZE + valid_len1],
                      ), dtype=np.float32)
    X_description_test1 = None
    gc.collect()
    print('-' * 50)
    X_test2 = hstack((X_dummies[TRAIN_SIZE + valid_len1:TRAIN_SIZE + valid_len2],
                      X_brand[TRAIN_SIZE + valid_len1:TRAIN_SIZE + valid_len2],
                      X_brand2[TRAIN_SIZE + valid_len1:TRAIN_SIZE + valid_len2],
                      X_category1[TRAIN_SIZE + valid_len1:TRAIN_SIZE + valid_len2],
                      X_category2[TRAIN_SIZE + valid_len1:TRAIN_SIZE + valid_len2],
                      X_category3[TRAIN_SIZE + valid_len1:TRAIN_SIZE + valid_len2],
                      X_category4[TRAIN_SIZE + valid_len1:TRAIN_SIZE + valid_len2],
                      X_hand_feature[TRAIN_SIZE + valid_len1:TRAIN_SIZE + valid_len2],
                      X_name[TRAIN_SIZE + valid_len1:TRAIN_SIZE + valid_len2],
                      X_description_test2,
                      X_gencat_cond[TRAIN_SIZE + valid_len1:TRAIN_SIZE + valid_len2],
                      X_subcat_1_cond[TRAIN_SIZE + valid_len1:TRAIN_SIZE + valid_len2],
                      X_subcat_2_cond[TRAIN_SIZE + valid_len1:TRAIN_SIZE + valid_len2],
                      ), dtype=np.float32)
    X_description_test2 = None
    gc.collect()
    print('-' * 50)
    X_test3 = hstack((X_dummies[TRAIN_SIZE + valid_len2:],
                      X_brand[TRAIN_SIZE + valid_len2:],
                      X_brand2[TRAIN_SIZE + valid_len2:],
                      X_category1[TRAIN_SIZE + valid_len2:],
                      X_category2[TRAIN_SIZE + valid_len2:],
                      X_category3[TRAIN_SIZE + valid_len2:],
                      X_category4[TRAIN_SIZE + valid_len2:],
                      X_hand_feature[TRAIN_SIZE + valid_len2:],
                      X_name[TRAIN_SIZE + valid_len2:],
                      X_description_test3,
                      X_gencat_cond[TRAIN_SIZE + valid_len2:],
                      X_subcat_1_cond[TRAIN_SIZE + valid_len2:],
                      X_subcat_2_cond[TRAIN_SIZE + valid_len2:],
                      ), dtype=np.float32)
    X_description_test3 = None
    gc.collect()

    print(X_dummies.shape, X_brand.shape, X_brand2.shape, X_category1.shape, X_category2.shape, X_category3.shape,
          X_category4.shape, X_gencat_cond.shape, X_subcat_1_cond.shape, X_subcat_2_cond.shape,
          X_hand_feature.shape, X_name.shape, X_train.shape, X_test1.shape, X_test2.shape, X_test3.shape)
    X_gencat_cond, X_subcat_1_cond, X_subcat_2_cond = None, None, None
    X_dummies, X_brand, X_brand2, X_category1, X_category2, X_category3, X_category4, X_hand_feature, X_name = None, None, None, None, None, None, None, None, None
    gc.collect()

    # csr_matrix
    X_train = X_train.tocsr()
    print('[{}] X_train completed.'.format(time.time() - start_time))
    X_test1 = X_test1.tocsr()
    print('[{}] X_test1 completed.'.format(time.time() - start_time))
    X_test2 = X_test2.tocsr()
    print('[{}] X_test2 completed.'.format(time.time() - start_time))
    X_test3 = X_test3.tocsr()
    print('[{}] X_test3 completed.'.format(time.time() - start_time))
    return X_train, X_test1, X_test2, X_test3


class vanila_conv1d_Regressor:
    def __init__(self, param_dict, Item_size):
        self.seed = 2018
        self.batch_size = int(param_dict['batch_size'] * 1.25)
        self.lr = param_dict['lr']

        name_seq_len = param_dict['name_Len']
        desc_seq_len = param_dict['description_Len']
        denselayer_units = param_dict['denselayer_units']
        embed_name = param_dict['embed_name']
        embed_desc = param_dict['embed_desc']
        embed_brand = param_dict['embed_brand']
        embed_cat_2 = param_dict['embed_cat_2']
        embed_cat_3 = param_dict['embed_cat_3']
        name_filter = param_dict['name_filter']
        desc_filter = param_dict['desc_filter']
        name_filter_size = param_dict['name_filter_size']
        desc_filter_size = param_dict['desc_filter_size']
        dense_drop = param_dict['dense_drop']

        name_voc_size = Item_size['name']
        desc_voc_size = Item_size['item_description']
        brand_voc_size = Item_size['brand_name']
        cat1_voc_size = Item_size['category_1']
        cat2_voc_size = Item_size['category_2']
        cat3_voc_size = Item_size['category_name']

        tf.reset_default_graph()
        graph = tf.Graph()
        graph.seed = self.seed
        with graph.as_default():
            self.place_name = tf.placeholder(tf.int32, shape=(None, name_seq_len))
            self.place_desc = tf.placeholder(tf.int32, shape=(None, desc_seq_len))
            self.place_brand = tf.placeholder(tf.int32, shape=(None, 1))
            self.place_cat1 = tf.placeholder(tf.int32, shape=(None, 1))
            self.place_cat2 = tf.placeholder(tf.int32, shape=(None, 1))
            self.place_cat3 = tf.placeholder(tf.int32, shape=(None, 1))
            self.place_ship = tf.placeholder(tf.float32, shape=(None, 1))
            self.place_cond = tf.placeholder(tf.uint8, shape=(None, 1))
            self.place_hand = tf.placeholder(tf.float32, shape=(None, len(Item_size['hand_feature'])))

            self.place_y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

            self.place_lr = tf.placeholder(tf.float32, shape=(), )

            self.is_train = tf.placeholder(tf.bool, shape=(), )

            name = embed(self.place_name, name_voc_size, embed_name)
            desc = embed(self.place_desc, desc_voc_size, embed_desc)
            brand = embed(self.place_brand, brand_voc_size, embed_brand)
            cat_2 = embed(self.place_cat2, cat2_voc_size, embed_cat_2)
            cat_3 = embed(self.place_cat3, cat3_voc_size, embed_cat_3)

            name = conv1d(name, num_filters=name_filter, filter_size=name_filter_size)
            name = tf.layers.average_pooling1d(name, pool_size=int(name.shape[1]), strides=1, padding='valid')
            name = tf.contrib.layers.flatten(name)

            desc = conv1d(desc, num_filters=desc_filter, filter_size=desc_filter_size)
            desc = tf.layers.average_pooling1d(desc, pool_size=int(desc.shape[1]), strides=1, padding='valid')
            desc = tf.contrib.layers.flatten(desc)

            brand = tf.contrib.layers.flatten(brand)

            cat_1 = tf.one_hot(self.place_cat1, cat1_voc_size)
            cat_1 = tf.contrib.layers.flatten(cat_1)

            cat_2 = tf.contrib.layers.flatten(cat_2)
            cat_3 = tf.contrib.layers.flatten(cat_3)

            hand_feat = self.place_hand
            ship = self.place_ship

            cond = tf.one_hot(self.place_cond, 5)
            cond = tf.contrib.layers.flatten(cond)

            out = tf.concat([name, desc, brand, cat_1, cat_2, cat_3, ship, cond, hand_feat], axis=1)
            out = dense(out, denselayer_units, activation=tf.nn.relu)
            out = tf.layers.dropout(out, rate=dense_drop, training=self.is_train)
            self.out = dense(out, 1)

            loss = tf.losses.mean_squared_error(self.place_y, self.out)
            opt = tf.train.AdamOptimizer(learning_rate=self.place_lr)
            self.train_step = opt.minimize(loss)

            init = tf.global_variables_initializer()

        config = tf.ConfigProto(intra_op_parallelism_threads=THREAD,
                                inter_op_parallelism_threads=THREAD,
                                allow_soft_placement=True, )
        self.session = tf.Session(config=config, graph=graph)
        self.init = init

    def fit(self, X_train, y_train, X_valid, VALID_SIZE):
        self.session.run(self.init)

        y_pred = np.zeros(VALID_SIZE)
        test_idx = np.arange(VALID_SIZE)
        test_batches = prepare_batches(test_idx, self.batch_size)

        total_batches = math.ceil(TRAIN_SIZE / self.batch_size)
        pred_epoch = total_batches - 1 - 400
        weight = 0.8

        for epoch in range(EPOCH):
            np.random.seed(epoch)

            train_idx_shuffle = np.arange(TRAIN_SIZE)
            np.random.shuffle(train_idx_shuffle)
            batches = prepare_batches(train_idx_shuffle, self.batch_size)

            for rnd, idx in enumerate(batches):
                feed_dict = {
                    self.place_name: X_train['X_seq_name'][idx],
                    self.place_desc: X_train['X_seq_item_description'][idx],
                    self.place_brand: X_train['X_brand_name'][idx],
                    self.place_cat1: X_train['X_category_1'][idx],
                    self.place_cat2: X_train['X_category_2'][idx],
                    self.place_cat3: X_train['X_category_name'][idx],
                    self.place_cond: X_train['X_item_condition_id'][idx],
                    self.place_ship: X_train['X_shipping'][idx],
                    self.place_hand: X_train['X_hand_feature'][idx],
                    self.place_y: y_train[idx],
                    self.place_lr: self.lr,
                    self.is_train: True,
                }
                self.session.run(self.train_step, feed_dict=feed_dict)

                if epoch == EPOCH - 1 and rnd == pred_epoch:
                    print(pred_epoch)
                    for idx in test_batches:
                        feed_dict = {
                            self.place_name: X_valid['X_seq_name'][idx],
                            self.place_desc: X_valid['X_seq_item_description'][idx],
                            self.place_brand: X_valid['X_brand_name'][idx],
                            self.place_cat1: X_valid['X_category_1'][idx],
                            self.place_cat2: X_valid['X_category_2'][idx],
                            self.place_cat3: X_valid['X_category_name'][idx],
                            self.place_cond: X_valid['X_item_condition_id'][idx],
                            self.place_ship: X_valid['X_shipping'][idx],
                            self.place_hand: X_valid['X_hand_feature'][idx],
                            self.is_train: False,
                        }
                        batch_pred = self.session.run(self.out, feed_dict=feed_dict)
                        y_pred[idx] += batch_pred[:, 0] * weight
                    pred_epoch += 200
                    weight += 0.2

        y_pred /= 3
        self.session.close()
        return y_pred


class vanila_embend_Regressor:
    def __init__(self, param_dict, Item_size):
        self.seed = 2018
        self.batch_size = int(param_dict['batch_size'] * 1.25)
        self.lr = param_dict['lr']

        name_seq_len = param_dict['name_Len']
        desc_seq_len = param_dict['description_Len']
        denselayer_units = param_dict['denselayer_units']
        embed_name = param_dict['embed_name']
        embed_desc = param_dict['embed_desc']
        embed_brand = param_dict['embed_brand']
        embed_cat_2 = param_dict['embed_cat_2']
        embed_cat_3 = param_dict['embed_cat_3']
        dense_drop = param_dict['dense_drop']

        name_voc_size = Item_size['name']
        desc_voc_size = Item_size['item_description']
        brand_voc_size = Item_size['brand_name']
        cat1_voc_size = Item_size['category_1']
        cat2_voc_size = Item_size['category_2']
        cat3_voc_size = Item_size['category_name']

        tf.reset_default_graph()
        graph = tf.Graph()
        graph.seed = self.seed
        with graph.as_default():
            self.place_name = tf.placeholder(tf.int32, shape=(None, name_seq_len))
            self.place_desc = tf.placeholder(tf.int32, shape=(None, desc_seq_len))
            self.place_brand = tf.placeholder(tf.int32, shape=(None, 1))
            self.place_cat1 = tf.placeholder(tf.int32, shape=(None, 1))
            self.place_cat2 = tf.placeholder(tf.int32, shape=(None, 1))
            self.place_cat3 = tf.placeholder(tf.int32, shape=(None, 1))
            self.place_ship = tf.placeholder(tf.float32, shape=(None, 1))
            self.place_cond = tf.placeholder(tf.uint8, shape=(None, 1))
            self.place_hand = tf.placeholder(tf.float32, shape=(None, len(Item_size['hand_feature'])))

            self.place_y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

            self.place_lr = tf.placeholder(tf.float32, shape=(), )

            self.is_train = tf.placeholder(tf.bool, shape=(), )

            name = embed(self.place_name, name_voc_size, embed_name)
            desc = embed(self.place_desc, desc_voc_size, embed_desc)
            brand = embed(self.place_brand, brand_voc_size, embed_brand)
            cat_2 = embed(self.place_cat2, cat2_voc_size, embed_cat_2)
            cat_3 = embed(self.place_cat3, cat3_voc_size, embed_cat_3)

            name = tf.layers.average_pooling1d(name, pool_size=int(name.shape[1]), strides=1, padding='valid')
            name = tf.contrib.layers.flatten(name)

            desc = tf.layers.average_pooling1d(desc, pool_size=int(desc.shape[1]), strides=1, padding='valid')
            desc = tf.contrib.layers.flatten(desc)

            brand = tf.contrib.layers.flatten(brand)

            cat_1 = tf.one_hot(self.place_cat1, cat1_voc_size)
            cat_1 = tf.contrib.layers.flatten(cat_1)

            cat_2 = tf.contrib.layers.flatten(cat_2)
            cat_3 = tf.contrib.layers.flatten(cat_3)

            hand_feat = self.place_hand
            ship = self.place_ship

            cond = tf.one_hot(self.place_cond, 5)
            cond = tf.contrib.layers.flatten(cond)

            out = tf.concat([name, desc, brand, cat_1, cat_2, cat_3, ship, cond, hand_feat], axis=1)
            out = dense(out, denselayer_units, activation=tf.nn.relu)
            out = tf.layers.dropout(out, rate=dense_drop, training=self.is_train)
            self.out = dense(out, 1)

            loss = tf.losses.mean_squared_error(self.place_y, self.out)
            opt = tf.train.AdamOptimizer(learning_rate=self.place_lr)
            self.train_step = opt.minimize(loss)

            init = tf.global_variables_initializer()

        config = tf.ConfigProto(intra_op_parallelism_threads=THREAD,
                                inter_op_parallelism_threads=THREAD,
                                allow_soft_placement=True, )
        self.session = tf.Session(config=config, graph=graph)
        self.init = init

    def fit(self, X_train, y_train, X_valid, VALID_SIZE):
        self.session.run(self.init)

        y_pred = np.zeros(VALID_SIZE)
        test_idx = np.arange(VALID_SIZE)
        test_batches = prepare_batches(test_idx, self.batch_size)

        total_batches = math.ceil(TRAIN_SIZE / self.batch_size)
        pred_epoch = total_batches - 1 - 400
        weight = [
            0.9116810454224276,
            0.9263363989954573,
            1.1619825555821153,
        ]
        pointer = 0

        for epoch in range(EPOCH):
            np.random.seed(epoch)

            train_idx_shuffle = np.arange(TRAIN_SIZE)
            np.random.shuffle(train_idx_shuffle)
            batches = prepare_batches(train_idx_shuffle, self.batch_size)

            for rnd, idx in enumerate(batches):
                feed_dict = {
                    self.place_name: X_train['X_seq_name'][idx],
                    self.place_desc: X_train['X_seq_item_description'][idx],
                    self.place_brand: X_train['X_brand_name'][idx],
                    self.place_cat1: X_train['X_category_1'][idx],
                    self.place_cat2: X_train['X_category_2'][idx],
                    self.place_cat3: X_train['X_category_name'][idx],
                    self.place_cond: X_train['X_item_condition_id'][idx],
                    self.place_ship: X_train['X_shipping'][idx],
                    self.place_hand: X_train['X_hand_feature'][idx],
                    self.place_y: y_train[idx],
                    self.place_lr: self.lr,
                    self.is_train: True,
                }
                self.session.run(self.train_step, feed_dict=feed_dict)

                if epoch == EPOCH - 1 and rnd == pred_epoch:
                    print(pred_epoch)
                    for idx in test_batches:
                        feed_dict = {
                            self.place_name: X_valid['X_seq_name'][idx],
                            self.place_desc: X_valid['X_seq_item_description'][idx],
                            self.place_brand: X_valid['X_brand_name'][idx],
                            self.place_cat1: X_valid['X_category_1'][idx],
                            self.place_cat2: X_valid['X_category_2'][idx],
                            self.place_cat3: X_valid['X_category_name'][idx],
                            self.place_cond: X_valid['X_item_condition_id'][idx],
                            self.place_ship: X_valid['X_shipping'][idx],
                            self.place_hand: X_valid['X_hand_feature'][idx],
                            self.is_train: False,
                        }
                        batch_pred = self.session.run(self.out, feed_dict=feed_dict)
                        y_pred[idx] += batch_pred[:, 0] * weight[pointer]
                    pred_epoch += 200
                    pointer += 1

        y_pred /= 3
        self.session.close()
        return y_pred


class vanila_FM_FTRL_Regressor:
    def __init__(self, param_dict, D):
        alpha = param_dict['alpha']
        beta = param_dict['beta']
        L1 = param_dict['L1']
        L2 = param_dict['L2']
        alpha_fm = param_dict['alpha_fm']
        init_fm = param_dict['init_fm']
        D_fm = param_dict['D_fm']
        e_noise = param_dict['e_noise']
        iters = param_dict['iters']

        self.model = FM_FTRL(alpha=alpha,
                             beta=beta,
                             L1=L1,
                             L2=L2,
                             D=D,
                             alpha_fm=alpha_fm,
                             L2_fm=0.0,
                             init_fm=init_fm,
                             D_fm=D_fm,
                             e_noise=e_noise,
                             iters=iters,
                             inv_link="identity",
                             threads=THREAD)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


def main():
    start_time = time.time()
    merge, submission, y_train = get_extract_feature()
    print('[{}] data preparation done.'.format(time.time() - start_time))

    if DEVELOP:
        VALID_SIZE = 10000
    else:
        VALID_SIZE = submission.shape[0]

    merge, Item_size, hand_feature = Preprocess_features(merge, start_time)
    Item_size['hand_feature'] = hand_feature

    X_train, X_test = Split_Train_Test_NN(merge, hand_feature)
    print('[{}] Split_Train_Test completed'.format(time.time() - start_time))

    print('[{}] training conv1d.'.format(time.time() - start_time))
    model = vanila_conv1d_Regressor(param_space_best_vanila_con1d, Item_size)
    predsConv1d = model.fit(X_train, y_train, X_test, VALID_SIZE).astype(np.float32)
    print('[{}] Train conv1d completed'.format(time.time() - start_time))

    print('[{}] training fasttext.'.format(time.time() - start_time))
    model = vanila_embend_Regressor(param_space_best_vanila_embend, Item_size)
    predsEmbend = model.fit(X_train, y_train, X_test, VALID_SIZE).astype(np.float32)
    print('[{}] Train fasttext completed'.format(time.time() - start_time))
    X_train, X_test, model, Item_size = None, None, None, None
    merge.drop(['Lb_category_1', 'Lb_category_2', 'Lb_category_name',
                'Lb_brand_name', 'Seq_item_description', 'Seq_name'],
                axis=1, inplace=True)
    gc.collect()

    X_train, X_test1, X_test2, X_test3 = Split_Train_Test_FTRL(merge, hand_feature, start_time)
    print('[{}] Split_Train_Test completed'.format(time.time() - start_time))
    merge, hand_feature = None, None
    gc.collect()

    print('[{}] training FM_FTRL.'.format(time.time() - start_time))
    model = vanila_FM_FTRL_Regressor(param_space_best_FM_FTRL, D=X_train.shape[1])
    X_train = X_train.astype(np.float64)
    y_train = y_train.reshape(-1).astype(np.float64)
    model.fit(X_train, y_train)
    print('[{}] Train FM_FTRL completed'.format(time.time() - start_time))
    X_train, y_train = None, None
    gc.collect()
    X_test1 = X_test1.astype(np.float64)
    predsFM_FTRL1 = model.predict(X_test1)
    X_test1 = None
    gc.collect()
    X_test2 = X_test2.astype(np.float64)
    predsFM_FTRL2 = model.predict(X_test2)
    X_test2 = None
    gc.collect()
    X_test3 = X_test3.astype(np.float64)
    predsFM_FTRL3 = model.predict(X_test3)
    X_test3 = None
    gc.collect()
    predsFM_FTRL = np.concatenate((predsFM_FTRL1, predsFM_FTRL2, predsFM_FTRL3))
    predsFM_FTRL1, predsFM_FTRL2, predsFM_FTRL3 = None, None, None
    print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))

    # preds = predsFM_FTRL
    preds = param_space_best_ensemble['FM_FTRL_weight'] * predsFM_FTRL + \
            param_space_best_ensemble['conv1d_weight'] * predsConv1d + \
            param_space_best_ensemble['embend_weight'] * predsEmbend

    submission['price'] = np.expm1(preds * PRICE_STD + PRICE_MEAN)
    print(submission['price'].mean())
    submission['price'] = submission['price'].apply(lambda x: max(x, 3.0))
    submission['price'] = submission['price'].apply(lambda x: min(x, 2000.0))
    submission.to_csv("submission_gru_lstm.csv", index=False)


if __name__ == "__main__":
    main()
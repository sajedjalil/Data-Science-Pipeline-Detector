import os
os.environ['OMP_NUM_THREADS'] = '4'

# ------------------------------
# global variables

SEED = 77
PATH_TRAIN = '../input/mercari-price-suggestion-challenge/train.tsv'
PATH_TEST = '../input/mercari-price-suggestion-challenge/test.tsv'
# PATH_TEST = '../input/test_debug.tsv'
# PATH_TEST = '../input/generate-simulated-test-data-for-2nd-stage/test_debug.tsv'
PATH_FASETEXT = '../input/fasttext-pretrained-word-vectors-english/wiki.en.bin'
PATH_WORD2VEC = '../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin'

IS_KERNEL = True
# IS_KERNEL = False

TRAIN_MINIMUM_VALUE = 0.0
PREDICT_MINIMUM_VALUE = 1e-10

# ------------------------------

os.environ['PYTHONHASHSEED'] = str(SEED)

import numpy as np
np.random.seed(SEED)

from multiprocessing import Pool

import time
import math
import re

import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers

import tensorflow as tf

from functools import partial

from nltk import tokenize
from nltk.stem.porter import PorterStemmer

from collections import Counter
from itertools import chain

from fastcache import clru_cache as lru_cache

import gc

# whether activate progress-bar
if IS_KERNEL:
    prog = lambda x: x
else:
    from tqdm import tqdm
    prog = tqdm

# set logger
from logging import getLogger, StreamHandler, Formatter
logger = getLogger('Logging')
logger.setLevel(20)
sh = StreamHandler()
logger.addHandler(sh)
formatter = Formatter('%(asctime)s: %(message)s')
sh.setFormatter(formatter)


# ------------------------------
# class
# ------------------------------

class MercariChainerNetwork(chainer.Chain):

    def __init__(self, sub_name_init_W, item_description_init_W, sub_categories_03_int_W):
        super(MercariChainerNetwork, self).__init__()
        with self.init_scope():
            self.embed_brand = L.EmbedID(1000, 15, ignore_label=-1)
            self.embed_name = L.EmbedID(100, 10, ignore_label=-1)
            self.sub_name = L.EmbedID(2000, 32, ignore_label=-1, initialW=sub_name_init_W)
            self.item_description = L.EmbedID(2000, 32, ignore_label=-1, initialW=item_description_init_W)
            self.embed_sub_category_01 = L.EmbedID(16, 8, ignore_label=-1)
            self.embed_sub_category_02 = L.EmbedID(128, 16, ignore_label=-1)
            self.embed_sub_category_03 = L.EmbedID(1024, 24, ignore_label=-1, initialW=sub_categories_03_int_W)

            self.fc1 = L.Linear(None, 128)
            self.fc2 = L.Linear(None, 128)
            self.fc3 = L.Linear(None, 1)

    def __call__(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output

    def feature_embed(self, data, sub_name_embed=None, item_description_embed=None):
        base = np.zeros((len(data), 14), dtype=np.float32)

        # ------------------------------
        # shipping

        base[:, 0] = data['shipping']

        # ------------------------------
        # item_condition_id

        base[:, 1] = data['item_condition_id'] == 1
        base[:, 2] = data['item_condition_id'] == 2
        base[:, 3] = data['item_condition_id'] == 3
        base[:, 4] = data['item_condition_id'] == 4
        base[:, 5] = data['item_condition_id'] == 5

        # ------------------------------
        # item_condition_id

        base[:, 6] = data['item_description'] == 'nodesc'

        base[:, 7] = data['item_description'].apply(len)
        base[data['item_description'] == 'nodesc', 7] = 0
        base[:, 7] = (base[:, 7] - 100.0) / 100.0

        # ------------------------------
        # same name features

        base[:, 8] = data['same_name_mean']
        base[:, 9] = data['same_name_count']
        base[:, 10] = data['same_brand_name_mean']
        base[:, 11] = data['same_brand_name_count']
        base[:, 12] = data['same_category_name_mean']
        base[:, 13] = data['same_category_name_count']

        # ------------------------------
        # embedding features

        target_brands = data['brand_name_index'].as_matrix()
        h_brand = self.embed_brand(target_brands)

        target_names = data['name_index'].as_matrix()
        h_name = self.embed_name(target_names)

        target_sub_categories_01 = data['sub_category_01_index'].as_matrix()
        h_sub_category_01 = self.embed_sub_category_01(target_sub_categories_01)

        target_sub_categories_02 = data['sub_category_02_index'].as_matrix()
        h_sub_category_02 = self.embed_sub_category_02(target_sub_categories_02)

        target_sub_categories_03 = data['sub_category_03_index'].as_matrix()
        h_sub_category_03 = self.embed_sub_category_03(target_sub_categories_03)

        # ------------------------------
        # name (sub)

        if sub_name_embed is None:
            sub_name_embed = F.vstack([self.sub_name(np.arange(2000, dtype=np.int32)),
                                       np.zeros((1, 32), dtype=np.float32)])

        data_sub_name_indexes = np.asarray(data['sub_name_indexes'].tolist(), dtype=np.int32)

        max_len = max(1, np.max(data['sub_name_len'].as_matrix()))
        # print('\n        maximum length of sub name: {}'.format(max_len))
        data_sub_name_indexes = data_sub_name_indexes[:, :max_len]

        h_sub_name_embed = F.get_item(sub_name_embed, data_sub_name_indexes.flatten())
        h_sub_name_reshaped = F.reshape(h_sub_name_embed, (len(data), -1, 32))
        h_sub_name = F.sum(h_sub_name_reshaped, axis=1)

        # ------------------------------
        # item_description name (sub)

        if item_description_embed is None:
            item_description_embed = F.vstack([self.item_description(np.arange(2000, dtype=np.int32)),
                                               np.zeros((1, 32), dtype=np.float32)])

        data_item_description_indexes = np.asarray(data['item_description_indexes'].tolist(), dtype=np.int32)

        max_len = max(1, np.max(data['item_description_len'].as_matrix()))
        # print('maximum length of item description: {}'.format(max_len))
        data_item_description_indexes = data_item_description_indexes[:, :max_len]

        h_item_description_embed = F.get_item(item_description_embed, data_item_description_indexes.flatten())
        h_item_description_reshaped = F.reshape(h_item_description_embed, (len(data), -1, 32))
        h_item_description = F.sum(h_item_description_reshaped, axis=1)

        # ------------------------------
        # merge

        merged = F.hstack([base, h_brand, h_name,
                           h_sub_category_01, h_sub_category_02, h_sub_category_03, h_sub_name, h_item_description])

        return merged


class MercariChainer:

    EPOCH = 7
    BATCH_SIZE = 1024

    def __init__(self):
        self.top_N_sub_categories_01 = None
        self.top_N_sub_categories_02 = None
        self.top_N_sub_categories_03 = None
        self.model = None

        self.name_to_indexes = None
        self.name_to_len = None

        self.item_description_to_indexes = None
        self.item_description_to_len = None

        self.top_N_categories = None
        self.top_N_brands = None
        self.top_N_names = None

        self.y_bias = 0.0
        self.y_scale = 1.0

    @staticmethod
    def get_word_vectors(num_words, *args):

        from pyfasttext import FastText
        word_model = FastText(PATH_FASETEXT)

        ret = list()
        for i, top_N_words in enumerate(args):

            init_W = np.zeros((num_words[i], 300))
            for c, n in enumerate(top_N_words):
                init_W[c, :] = word_model.get_numpy_vector(n)

            ret.append(init_W)

        return tuple(ret)

    @staticmethod
    def reduct_dims_ica(data, n_components):
        ica = FastICA(n_components=n_components, max_iter=1000)
        ica.fit(data.T)
        return ica.components_.T

    @staticmethod
    def convert_to_indexes(s, top_N_dict, num_elements):
        indexes_list = [top_N_dict[k] for k in tokenize.wordpunct_tokenize(s) if k in top_N_dict]
        indexes_list.extend([-1] * num_elements)
        indexes_list = indexes_list[:num_elements]
        return indexes_list

    @staticmethod
    def convert_to_len(s, top_N_dict, num_elements):
        indexes_list = [top_N_dict[k] for k in tokenize.wordpunct_tokenize(s) if k in top_N_dict]
        return min(num_elements, len(indexes_list))

    @staticmethod
    def get_index_element(s, index):
        splitted = s.split('/')
        if len(splitted) > index:
            return s.split('/')[index]
        else:
            return 'category_NaN'

    @staticmethod
    def util_clean_name(s):
        s = s.lower()
        s = re.sub(re.compile("(\(|\)|!|\.|,)"), '', s)
        return s

    @staticmethod
    def util_collect_top_n_words(clean_series, n=2000):

        def util_concat_list(l):
            ret = list()
            for sub in l:
                ret.extend(sub)
            return ret

        name_split = clean_series.apply(tokenize.wordpunct_tokenize).tolist()
        name_flatten = util_concat_list(name_split)
        top_n_words = pd.Series(name_flatten).value_counts().index[:n].tolist()

        return top_n_words

    def training(self, train):

        train.sort_values(by='item_description_len', inplace=True)
        train.reset_index(drop=True, inplace=True)

        optimizer = optimizers.Adam(alpha=2e-3)
        optimizer.setup(self.model)

        train_price_log1p = np.maximum(train[['price']].as_matrix(), TRAIN_MINIMUM_VALUE)
        train_price_log1p = np.log1p(train_price_log1p).astype(np.float32)

        self.y_bias = np.mean(train_price_log1p)
        self.y_scale = np.std(train_price_log1p - self.y_bias)

        index_start_points = np.arange(0, len(train), self.BATCH_SIZE)

        for ep in range(self.EPOCH):

            logger.info('[c] epoch: {}'.format(ep))

            if ep >= self.EPOCH - 2:
                optimizer.alpha *= 0.1
                logger.info('[c] update optimizer alpha: {0:.6f}'.format(optimizer.alpha))

            np.random.shuffle(index_start_points)

            loss_value = 0

            for c, i in prog(enumerate(index_start_points)):

                input_batch = train.iloc[i:i + self.BATCH_SIZE, :]
                merged = self.model.feature_embed(input_batch)
                output = self.model(merged)

                target = train_price_log1p[i:i + self.BATCH_SIZE, :]
                target = (target - self.y_bias) / self.y_scale

                loss = F.mean_squared_error(output, target)

                self.model.cleargrads()
                loss.backward()
                optimizer.update()

                loss_value += loss.data * len(merged)

            train_score = math.sqrt(loss_value / len(train))
            logger.info('[c] train loss: {0:.4f}'.format(train_score))

    def fit_preprocess(self, train):

        # ------------------------------
        # top N keywords

        # sub category
        logger.info('[c] check frequent word in category_name')
        train_sub_category_name_03 = train['category_name'].apply(lambda s: self.get_index_element(s, 2))
        self.top_N_sub_categories_03 = train_sub_category_name_03.value_counts().index[:1024].tolist()

        # name
        logger.info('[c] check frequent word in name')
        train_name_clean = train['name'].apply(self.util_clean_name)
        top_N_sub_names = self.util_collect_top_n_words(train_name_clean, n=2000)

        # item_description
        logger.info('[c] check frequent word in item_description')
        train_item_description_clean = train['item_description'].apply(self.util_clean_name)
        top_N_item_description = self.util_collect_top_n_words(train_item_description_clean, n=2000)

        logger.info('[c] get word vectors from pre-train fast-text')
        sub_name_init_W, item_description_init_W, sub_categories_03_init_W = self.get_word_vectors(
            [2000, 2000, 1024],
            top_N_sub_names, top_N_item_description, self.top_N_sub_categories_03)

        logger.info('[c] reduce dimensions for network input')
        sub_name_init_W = self.reduct_dims_ica(sub_name_init_W, n_components=32)
        item_description_init_W = self.reduct_dims_ica(item_description_init_W, n_components=32)
        sub_categories_03_init_W = self.reduct_dims_ica(sub_categories_03_init_W, n_components=24)

        # model
        logger.info('[c] initialize deep model')
        self.model = MercariChainerNetwork(sub_name_init_W, item_description_init_W, sub_categories_03_init_W)

        logger.info('[c] encode name to index-sequences')
        top_N_sub_names_dict = {key: value for (value, key) in enumerate(top_N_sub_names)}

        self.name_to_indexes = partial(self.convert_to_indexes, top_N_dict=top_N_sub_names_dict, num_elements=20)
        self.name_to_len = partial(self.convert_to_len, top_N_dict=top_N_sub_names_dict, num_elements=20)

        train.loc[:, 'sub_name_indexes'] = train_name_clean.apply(self.name_to_indexes)
        train.loc[:, 'sub_name_len'] = train_name_clean.apply(self.name_to_len)

        logger.info('[c] encode item_description to index-sequences')
        top_N_item_description_dict = {key: value for (value, key) in enumerate(top_N_item_description)}

        self.item_description_to_indexes = partial(self.convert_to_indexes,
                                                   top_N_dict=top_N_item_description_dict, num_elements=100)

        self.item_description_to_len = partial(self.convert_to_len,
                                               top_N_dict=top_N_item_description_dict, num_elements=100)

        train.loc[:, 'item_description_indexes'] = train_item_description_clean.apply(self.item_description_to_indexes)
        train.loc[:, 'item_description_len'] = train_item_description_clean.apply(self.item_description_to_len)

        logger.info('[c] split category_name into sub_category')
        train_sub_category_name_01 = train['category_name'].apply(lambda s: self.get_index_element(s, 0))
        train_sub_category_name_02 = train['category_name'].apply(lambda s: self.get_index_element(s, 1))

        self.top_N_sub_categories_01 = train_sub_category_name_01.value_counts().index[:16].tolist()
        self.top_N_sub_categories_02 = train_sub_category_name_02.value_counts().index[:128].tolist()

        logger.info('[c] encode sub_category to indexes')
        train.loc[:, 'sub_category_01_index'] = pd.Categorical(train_sub_category_name_01,
                                                               categories=self.top_N_sub_categories_01).codes
        train.loc[:, 'sub_category_02_index'] = pd.Categorical(train_sub_category_name_02,
                                                               categories=self.top_N_sub_categories_02).codes
        train.loc[:, 'sub_category_03_index'] = pd.Categorical(train_sub_category_name_03,
                                                               categories=self.top_N_sub_categories_03).codes

        logger.info('[c] encode popular category_name to index')
        self.top_N_categories = train['category_name'].value_counts().index[:1000].tolist()
        train.loc[:, 'category_name_index'] = pd.Categorical(train['category_name'],
                                                             categories=self.top_N_categories).codes

        logger.info('[c] encode popular brand_name to index')
        self.top_N_brands = train['brand_name'].value_counts().index[:1000].tolist()
        train.loc[:, 'brand_name_index'] = pd.Categorical(train['brand_name'], categories=self.top_N_brands).codes

        logger.info('[c] encode popular name to index')
        self.top_N_names = train['name'].value_counts().index[:100].tolist()
        train.loc[:, 'name_index'] = pd.Categorical(train['name'], categories=self.top_N_names).codes

        return train

    def fit(self, train):

        df_train = self.fit_preprocess(train)
        gc.collect()
        self.training(df_train)

    def predict_nn(self, data):

        result = np.zeros(len(data), dtype=np.float32)

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            sub_name_embed = F.vstack([self.model.sub_name(np.arange(2000, dtype=np.int32)),
                                       np.zeros((1, 32), dtype=np.float32)])
            item_description_embed = F.vstack([self.model.item_description(np.arange(2000, dtype=np.int32)),
                                               np.zeros((1, 32), dtype=np.float32)])

        for i in range(0, len(data), self.BATCH_SIZE):
            target_data = data.iloc[i:i + self.BATCH_SIZE, :]

            input = self.model.feature_embed(target_data,
                                             sub_name_embed=sub_name_embed,
                                             item_description_embed=item_description_embed)

            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                output = self.model(input)

            result[i:i + self.BATCH_SIZE] = np.expm1(output.data.flatten() * self.y_scale + self.y_bias)

        return result

    def predict(self, data):

        data_name_clean = data['name'].apply(self.util_clean_name)

        logger.debug('[c] encode name to index-sequences')
        data.loc[:, 'sub_name_indexes'] = data_name_clean.apply(self.name_to_indexes)
        data.loc[:, 'sub_name_len'] = data_name_clean.apply(self.name_to_len)

        logger.debug('[c] encode item_description to index-sequences')
        data_item_description_clean = data['item_description'].apply(self.util_clean_name)

        data.loc[:, 'item_description_indexes'] = data_item_description_clean.apply(self.item_description_to_indexes)
        data.loc[:, 'item_description_len'] = data_item_description_clean.apply(self.item_description_to_len)

        logger.debug('[c] split category_name into sub_category')
        data_sub_category_name_01 = data['category_name'].apply(lambda s: self.get_index_element(s, 0))
        data_sub_category_name_02 = data['category_name'].apply(lambda s: self.get_index_element(s, 1))
        data_sub_category_name_03 = data['category_name'].apply(lambda s: self.get_index_element(s, 2))

        logger.debug('[c] encode sub_category to indexes')
        data.loc[:, 'sub_category_01_index'] = pd.Categorical(data_sub_category_name_01,
                                                              categories=self.top_N_sub_categories_01).codes
        data.loc[:, 'sub_category_02_index'] = pd.Categorical(data_sub_category_name_02,
                                                              categories=self.top_N_sub_categories_02).codes
        data.loc[:, 'sub_category_03_index'] = pd.Categorical(data_sub_category_name_03,
                                                              categories=self.top_N_sub_categories_03).codes

        logger.debug('[c] encode popular category_name to index')
        data.loc[:, 'category_name_index'] = pd.Categorical(data['category_name'],
                                                            categories=self.top_N_categories).codes

        logger.debug('[c] encode popular brand_name to index')
        data.loc[:, 'brand_name_index'] = pd.Categorical(data['brand_name'], categories=self.top_N_brands).codes

        logger.debug('[c] encode popular name to index')
        data.loc[:, 'name_index'] = pd.Categorical(data['name'], categories=self.top_N_names).codes

        # sort by length of item_description, which makes inference faster
        data.sort_values(by='item_description_len', inplace=True)

        data.loc[:, 'test_predict'] = self.predict_nn(data)

        # sort by original index to make a submission
        data.sort_index(inplace=True)

        data_predict = data.loc[:, 'test_predict'].as_matrix()
        data_predict = data_predict.astype(np.float32)

        return np.maximum(data_predict, PREDICT_MINIMUM_VALUE)


class MercatiTFTokenizer:

    def __init__(self, min_df=10, tokenizer=str.split, max_len=None):
        self.min_df = min_df
        self.tokenizer = tokenizer
        self.doc_freq = None
        self.vocab = None
        self.vocab_idx = None
        self.max_len = max_len

    def fit_transform(self, texts):
        tokenized = []

        n = len(texts)

        for text in texts:
            sentence = self.tokenizer(text)
            tokenized.append(sentence)

        doc_freq = Counter(chain.from_iterable(tokenized))

        vocab = sorted([t for (t, c) in doc_freq.items() if c >= self.min_df])
        vocab_idx = {t: (i + 1) for (i, t) in enumerate(vocab)}
        doc_freq = [doc_freq[t] for t in vocab]

        self.doc_freq = doc_freq
        self.vocab = vocab
        self.vocab_idx = vocab_idx

        logger.info('[t] length of vocab {}'.format(len(vocab)))

        max_len_train = 0
        result_list = []
        for text in tokenized:
            text = self.text_to_idx(text)
            max_len_train = max(max_len_train, len(text))
            result_list.append(text)

        if self.max_len is None:
            self.max_len = max_len_train

        logger.info('[t] max_len is set as {}'.format(self.max_len))
        result = np.zeros(shape=(n, self.max_len), dtype=np.uint16)
        for i in range(n):
            text = result_list[i]
            result[i, :len(text)] = text[:self.max_len]

        return result

    def text_to_idx(self, tokenized):
        return [self.vocab_idx[t] for t in tokenized if t in self.vocab_idx]

    def transform(self, texts):
        n = len(texts)
        result = np.zeros(shape=(n, self.max_len), dtype=np.uint16)

        for i in range(n):
            text = self.tokenizer(texts[i])
            text = self.text_to_idx(text)
            result[i, :len(text)] = text[:self.max_len]

        return result

    def vocabulary_size(self):
        return len(self.vocab) + 1


class MercariTF_init:

    non_letter = re.compile(r'\W+')
    whitespace = re.compile(r'\s+')
    stemmer = PorterStemmer()

    DESC_NUM_COL = 52

    def __init__(self):
        self.name_tok = MercatiTFTokenizer(min_df=10, tokenizer=self.tokenize)
        self.desc_tok = MercatiTFTokenizer(min_df=50, tokenizer=self.tokenize, max_len=self.DESC_NUM_COL)

    @classmethod
    @lru_cache(2048)
    def stem(cls, s):
        return cls.stemmer.stem(s)

    @classmethod
    def tokenize(cls, text):
        text = text.lower()
        text = cls.non_letter.sub(' ', text)
        tokens = [cls.stem(t) for t in text.split()]

        return tokens

    @staticmethod
    def reduct_dims_ica(data, n_components):
        ica = FastICA(n_components=n_components, max_iter=1000)
        ica.fit(data.T)
        return ica.components_.T

    @staticmethod
    def get_word_vectors_word2vec(*args):

        ret = list()

        import gensim
        word_model = gensim.models.KeyedVectors.load_word2vec_format(PATH_WORD2VEC, binary=True)

        for top_N_words in args:

            init_W = np.random.normal(0.0, 0.25, (len(top_N_words) + 1, 300))

            not_found = 0
            found = 0

            for c, n in enumerate(top_N_words):
                try:
                    init_W[c + 1, :] = word_model[n]
                    found += 1
                except KeyError:
                    # print('not found: {}'.format(n))
                    not_found += 1

            logger.info('[p] word2vec found: {}  not-found: {}'.format(found, not_found))

            ret.append(init_W)

        return tuple(ret)

    def fit(self, df_train):

        tic = time.time()

        logger.info('[t] processing title')

        X_name = self.name_tok.fit_transform(df_train.name)

        logger.info('[t] processing description')

        X_desc = self.desc_tok.fit_transform(df_train.item_description)

        logger.info('[t] defining embedding dimension')

        name_embeddings_dim = 32
        desc_embeddings_dim = 32

        logger.info('[t] load pre-trained word vectors')

        name_init_W, desc_init_W = self.get_word_vectors_word2vec(self.name_tok.vocab, self.desc_tok.vocab)
        name_init_W = self.reduct_dims_ica(name_init_W, name_embeddings_dim).astype(np.float32)
        desc_init_W = self.reduct_dims_ica(desc_init_W, desc_embeddings_dim).astype(np.float32)

        time_pre_tf = (time.time() - tic) / 60.0
        logger.info('[t] elapsed time (preprocess for tf) : {0:.1f} [min]'.format(time_pre_tf))

        return self.name_tok, self.desc_tok, name_init_W, desc_init_W, X_name, X_desc


class MercariTF:

    non_letter = re.compile(r'\W+')
    whitespace = re.compile(r'\s+')

    EPOCH = 6

    def __init__(self, name_tok, desc_tok, name_W, desc_W, X_name, X_desc):

        self.cat_tok = MercatiTFTokenizer(min_df=55)
        self.name_tok = name_tok
        self.desc_tok = desc_tok

        self.y_bias = 0.0
        self.y_scale = 1.0

        # placeholders
        self.place_name = None
        self.place_desc = None
        self.place_brand = None
        self.place_cat = None
        self.place_ship = None
        self.place_cond = None

        # tensor flow session
        self.session = None

        self.name_init_W = name_W
        self.desc_init_W = desc_W
        self.X_name = X_name
        self.X_desc = X_desc

    @classmethod
    @lru_cache(1024)
    def cat_process(cls, cat):
        cat = cat.lower()
        cat = cls.whitespace.sub('', cat)
        split = cat.split('/')
        return cls.paths(split)

    def fit(self, df_train):

        logger.info('[t] reading train data')

        y = np.maximum(df_train['price'].as_matrix(), TRAIN_MINIMUM_VALUE)
        y = np.log1p(y).astype(np.float32)

        self.y_bias = np.mean(y)
        self.y_scale = np.std(y)
        y = (y - self.y_bias) / self.y_scale
        y = y.reshape(-1, 1)

        logger.info('[t] processing category')

        df_train_category_name_tree = df_train.category_name.apply(self.cat_process)
        X_cat = self.cat_tok.fit_transform(df_train_category_name_tree)
        cat_voc_size = self.cat_tok.vocabulary_size()

        logger.info('[t] processing title')

        X_name = self.X_name
        name_voc_size = self.name_tok.vocabulary_size()

        logger.info('[t] processing description')

        X_desc = self.X_desc
        desc_voc_size = self.desc_tok.vocabulary_size()

        logger.info('[t] defining embedding dimension')

        name_embeddings_dim = 32
        desc_embeddings_dim = 32
        brand_embeddings_dim = 4
        cat_embeddings_dim = 14

        logger.info('[t] load pre-trained word vectors')

        name_init_W = self.name_init_W
        desc_init_W = self.desc_init_W

        logger.info('[t] processing brand')

        df_train_brand_name = df_train.brand_name.str.lower()
        df_train_brand_name = df_train_brand_name.str.replace(' ', '_')

        brand_cnt = Counter(df_train_brand_name[df_train_brand_name != 'unknown'])
        brands = sorted(b for (b, c) in brand_cnt.items() if c >= 50)
        self.brands_idx = {b: (i + 1) for (i, b) in enumerate(brands)}

        X_brand = df_train_brand_name.apply(lambda b: self.brands_idx.get(b, 0))
        X_brand = X_brand.values.reshape(-1, 1)
        brand_voc_size = len(brands) + 1

        logger.info('[t] processing other features')

        X_item_cond = (df_train.item_condition_id - 1).astype('uint8').values.reshape(-1, 1)
        X_shipping = df_train.shipping.astype('float32').values.reshape(-1, 1)

        logger.info('[t] defining the model')

        name_seq_len = X_name.shape[1]
        desc_seq_len = X_desc.shape[1]
        cat_seq_len = X_cat.shape[1]

        graph = tf.Graph()
        graph.seed = SEED

        with graph.as_default():

            tf.set_random_seed(SEED)

            self.place_name = tf.placeholder(tf.int32, shape=(None, name_seq_len))
            self.place_desc = tf.placeholder(tf.int32, shape=(None, desc_seq_len))
            self.place_brand = tf.placeholder(tf.int32, shape=(None, 1))
            self.place_cat = tf.placeholder(tf.int32, shape=(None, cat_seq_len))
            self.place_ship = tf.placeholder(tf.float32, shape=(None, 1))
            self.place_cond = tf.placeholder(tf.uint8, shape=(None, 1))

            place_y = tf.placeholder(dtype=tf.float32, shape=(None, 1))
            place_lr = tf.placeholder(tf.float32, shape=(), )

            name = self.embed(self.place_name, name_voc_size, name_embeddings_dim, weight=name_init_W)
            desc = self.embed(self.place_desc, desc_voc_size, desc_embeddings_dim, weight=desc_init_W)
            brand = self.embed(self.place_brand, brand_voc_size, brand_embeddings_dim)
            cat = self.embed(self.place_cat, cat_voc_size, cat_embeddings_dim)

            name = self.conv1d(name, num_filters=16, filter_size=3)
            name = tf.contrib.layers.flatten(name)
            logger.info('[t] name shape: {}'.format(name.shape))

            desc = self.conv1d(desc, num_filters=16, filter_size=3)
            desc = tf.contrib.layers.flatten(desc)
            logger.info('[t] desc shape: {}'.format(desc.shape))

            brand = tf.contrib.layers.flatten(brand)
            logger.info('[t] brand shape: {}'.format(brand.shape))

            cat = tf.layers.average_pooling1d(cat, pool_size=cat_seq_len, strides=1, padding='valid')
            cat = tf.contrib.layers.flatten(cat)
            logger.info('[t] cat shape: {}'.format(cat.shape))

            ship = self.place_ship
            logger.info('[t] ship shape: {}'.format(ship.shape))

            cond = tf.one_hot(self.place_cond, 5)
            cond = tf.contrib.layers.flatten(cond)
            logger.info('[t] cond shape: {}'.format(cond.shape))

            out = tf.concat([name, desc, brand, cat, ship, cond], axis=1)
            logger.info('[t] concatenated shape: {}'.format(out.shape))

            out = self.dense(out, 100, activation=tf.nn.relu)
            self.out = self.dense(out, 1)

            loss = tf.losses.mean_squared_error(place_y, self.out)
            opt = tf.train.AdamOptimizer(learning_rate=place_lr)
            train_step = opt.minimize(loss)

            init = tf.global_variables_initializer()

        session_conf = tf.ConfigProto(intra_op_parallelism_threads=5, inter_op_parallelism_threads=1)
        self.session = tf.Session(config=session_conf, graph=graph)
        self.session.run(init)

        logger.info('[t] training the model')

        for i in range(self.EPOCH):
            t0 = time.time()
            np.random.seed(i)
            train_idx_shuffle = np.arange(X_name.shape[0])
            np.random.shuffle(train_idx_shuffle)
            batches = self.prepare_batches(train_idx_shuffle, 500)

            if i <= 2:
                lr = 0.001
            else:
                lr = 0.0001

            for idx in batches:
                feed_dict = {
                    self.place_name: X_name[idx],
                    self.place_desc: X_desc[idx],
                    self.place_brand: X_brand[idx],
                    self.place_cat: X_cat[idx],
                    self.place_cond: X_item_cond[idx],
                    self.place_ship: X_shipping[idx],
                    place_y: y[idx],
                    place_lr: lr,
                }
                self.session.run(train_step, feed_dict=feed_dict)

            took = time.time() - t0
            logger.info('[t] epoch %d took %.3fs' % (i, took))

    def predict(self, df_data):

        df_data.category_name = df_data.category_name.apply(self.cat_process)
        df_data.brand_name = df_data.brand_name.str.lower()
        df_data.brand_name = df_data.brand_name.str.replace(' ', '_')

        X_cat_data = self.cat_tok.transform(df_data.category_name)
        X_name_data = self.name_tok.transform(df_data.name)
        X_desc_data = self.desc_tok.transform(df_data.item_description)

        X_item_cond_data = (df_data.item_condition_id - 1).astype('uint8').values.reshape(-1, 1)
        X_shipping_data = df_data.shipping.astype('float32').values.reshape(-1, 1)

        X_brand_data = df_data.brand_name.apply(lambda b: self.brands_idx.get(b, 0))
        X_brand_data = X_brand_data.values.reshape(-1, 1)

        n_data = len(df_data)
        y_pred = np.zeros(n_data)

        test_idx = np.arange(n_data)
        batches = self.prepare_batches(test_idx, 5000)

        for idx in batches:
            feed_dict = {
                self.place_name: X_name_data[idx].astype(np.int32),
                self.place_desc: X_desc_data[idx].astype(np.int32),
                self.place_brand: X_brand_data[idx],
                self.place_cat: X_cat_data[idx].astype(np.int32),
                self.place_cond: X_item_cond_data[idx],
                self.place_ship: X_shipping_data[idx],
            }
            batch_pred = self.session.run(self.out, feed_dict=feed_dict)
            y_pred[idx] = batch_pred[:, 0]

        y_pred = y_pred * self.y_scale + self.y_bias
        data_predict = np.expm1(y_pred)

        data_predict = data_predict.astype(np.float32)

        return np.maximum(data_predict, PREDICT_MINIMUM_VALUE)

    @staticmethod
    def paths(tokens):
        all_paths = ['/'.join(tokens[0:(i + 1)]) for i in range(len(tokens))]
        return ' '.join(all_paths)

    @staticmethod
    def prepare_batches(seq, step):
        n = len(seq)
        res = []
        for i in range(0, n, step):
            res.append(seq[i:i + step])
        return res

    @staticmethod
    def conv1d(inputs, num_filters, filter_size, padding='same'):
        he_std = np.sqrt(2 / (filter_size * num_filters))
        out = tf.layers.conv1d(
            inputs=inputs, filters=num_filters, padding=padding,
            kernel_size=filter_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(stddev=he_std))
        return out

    @staticmethod
    def dense(X, size, reg=0.0, activation=None):
        he_std = np.sqrt(2 / int(X.shape[1]))
        out = tf.layers.dense(X, units=size, activation=activation,
                              kernel_initializer=tf.random_normal_initializer(stddev=he_std),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))
        return out

    @staticmethod
    def embed(inputs, size, dim, weight=None):
        std = np.sqrt(2 / dim)
        if weight is None:
            emb = tf.Variable(tf.random_uniform([size, dim], -std, std))
        else:
            emb = tf.Variable(weight)
        lookup = tf.nn.embedding_lookup(emb, inputs)
        return lookup


class MercariRidge:

    ITER_LGBM = 1800
    ITER_RIDGE = 250
    ITER_FM = 15

    def __init__(self):
        # name (use CountVectorizer)
        self.count_name = CountVectorizer(ngram_range=(1, 2))

        # transform category_name (use CountVectorizer)
        self.count_category_name = CountVectorizer()

        # brand_name (use LabelBinarizer)
        self.pop_brands = None
        self.vect_brand = LabelBinarizer(sparse_output=True)

        # item_condition_id
        self.ic_categories = None

        # shipping
        self.sp_categories = None

        # item description (use TfidfVectorizer)
        self.count_description = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=50000)

        # models
        from lightgbm import LGBMRegressor
        lgbm_params = {'n_estimators': self.ITER_LGBM,
                       'learning_rate': 1.0,
                       'max_depth': 5,
                       'num_leaves': 32,
                       'subsample': 0.9,
                       'colsample_bytree': 0.8,
                       'min_child_samples': 20,
                       'n_jobs': 4,
                       'max_bin': 31,
                       'random_state': SEED,
                       'verbose': -1}

        self.model_lgbm = LGBMRegressor(**lgbm_params)

        from sklearn.linear_model import Ridge
        self.model_ridge = Ridge(solver='sag', fit_intercept=True, max_iter=self.ITER_RIDGE)

        from fastFM import als
        self.model_fm = als.FMRegression(n_iter=self.ITER_FM, rank=8, l2_reg_w=10.0, l2_reg_V=15.0, random_state=SEED)

    def fit_preprocess(self, train):

        X_same_name_train = train[['same_name_mean', 'same_name_count']].as_matrix()

        logger.info('[r] count name')
        X_name_train = self.count_name.fit_transform(train['name'])

        logger.info('[r] count category_name')
        X_category_name_train = self.count_category_name.fit_transform(train['category_name'])

        logger.info('[r] binalize label with brand_name')
        self.pop_brands = train['brand_name'].value_counts().index[:2500]
        train_brand_name = train['brand_name'].copy()
        train_brand_name[~train_brand_name.isin(self.pop_brands)] = 'Other'
        X_brand_name_train = self.vect_brand.fit_transform(train_brand_name)

        logger.info('[r] encode item_condition_id')
        self.ic_categories = np.unique(train['item_condition_id'])
        train_item_condition_id = train['item_condition_id'].astype('category', categories=self.ic_categories)
        X_item_condition_id_train = csr_matrix(pd.get_dummies(train_item_condition_id, sparse=True))

        logger.info('[r] encode shipping')
        self.sp_categories = np.unique(train['shipping'])
        train_shipping = train['shipping'].astype('category', categories=self.sp_categories)
        X_shipping_train = csr_matrix(pd.get_dummies(train_shipping, sparse=True))

        logger.info('[r] encode item_description')
        X_item_description_train = self.count_description.fit_transform(train['item_description'])

        logger.info('[r] hstack all columns')
        X_train = hstack([X_name_train, X_brand_name_train, X_shipping_train, X_item_condition_id_train,
                          X_category_name_train, X_item_description_train, X_same_name_train]).tocsr()

        return X_train

    def fit(self, train):

        X_train = self.fit_preprocess(train)

        y_train = np.maximum(train['price'].as_matrix(), TRAIN_MINIMUM_VALUE)
        y_train = np.log1p(y_train).astype(np.float32)

        logger.info('[r] fit lgbm model......')
        self.model_lgbm.fit(X_train, y_train, eval_metric=self.rmsle_lgb)

        logger.info('[r] predict train by lgbm')
        preds_train_lgbm = self.model_lgbm.predict(X_train)

        logger.info('[r] fit fm model......')
        self.model_fm.fit(X_train, y_train)

        X_train = hstack([X_train, preds_train_lgbm.reshape(-1, 1)]).tocsr()

        logger.info('[r] fit ridge model......')
        self.model_ridge.fit(X_train, y_train)

        logger.info('[r] actual number of iterations: {}'.format(self.model_ridge.n_iter_))

    @staticmethod
    def rmsle_lgb(labels, preds):
        return 'rmsle', calc_valid_score(preds, labels), False

    def predict(self, data):

        X_same_name = data[['same_name_mean', 'same_name_count']].as_matrix()

        logger.debug('[r] count name')
        X_name = self.count_name.transform(data['name'])

        logger.debug('[r] count category_name')
        X_category_name = self.count_category_name.transform(data['category_name'])

        logger.debug('[r] binalize label with brand_name')
        data_brand_name = data['brand_name'].copy()
        data_brand_name[~data_brand_name.isin(self.pop_brands)] = 'Other'
        X_brand_name = self.vect_brand.transform(data_brand_name)

        logger.debug('[r] encode item_condition_id')
        data_item_condition_id = data['item_condition_id'].astype('category', categories=self.ic_categories)
        X_item_condition_id = csr_matrix(pd.get_dummies(data_item_condition_id, sparse=True))

        logger.debug('[r] encode shipping')
        data_shipping = data['shipping'].astype('category', categories=self.sp_categories)
        X_shipping = csr_matrix(pd.get_dummies(data_shipping, sparse=True))

        logger.debug('[r] encode item_description')
        X_item_description = self.count_description.transform(data['item_description'])

        logger.debug('[r] hstack all columns')
        X_data = hstack([X_name, X_brand_name, X_shipping, X_item_condition_id,
                         X_category_name, X_item_description, X_same_name]).tocsr()

        preds_data_lgbm = self.model_lgbm.predict(X_data)
        preds_data_fm = self.model_fm.predict(X_data)

        X_data = hstack([X_data, preds_data_lgbm.reshape(-1, 1)]).tocsr()
        preds_data_ridge = self.model_ridge.predict(X_data)

        preds_data_fm = np.expm1(preds_data_fm).astype(np.float32)
        preds_data_ridge = np.expm1(preds_data_ridge).astype(np.float32)

        return np.maximum(preds_data_ridge, PREDICT_MINIMUM_VALUE), np.maximum(preds_data_fm, PREDICT_MINIMUM_VALUE)


class SameValueFeatures:

    def __init__(self, target):
        self.target = target
        self.means = None

        self.bias_log_same_mean = None
        self.scale_log_same_mean = None

        self.bias_log_same_count = None
        self.scale_log_same_count = None

        self.fname_mean = 'same_{}_mean'.format(self.target)
        self.fname_count = 'same_{}_count'.format(self.target)

    def normalize(self, data):

        data[self.fname_mean] = (data[self.fname_mean] - self.bias_log_same_mean) / self.scale_log_same_mean
        data[self.fname_count] = (data[self.fname_count] - self.bias_log_same_count) / self.scale_log_same_count

        data[self.fname_mean].fillna(0.0, inplace=True)

    def fit_transform(self, train):

        self.means = train.groupby([self.target])['price'].agg([np.sum, len])
        self.means[self.target] = self.means.index

        train_merged = pd.merge(train, self.means, how='left')

        train_same_name_count = train_merged['len'] - 1
        train_same_name_mean = (train_merged['sum'] - train_merged['price']) / train_same_name_count

        train_same_name_count.fillna(0.0, inplace=True)

        train[self.fname_mean] = train_same_name_mean
        train[self.fname_count] = np.log1p(train_same_name_count)

        # ------------------------------
        # define scale and bias

        self.bias_log_same_mean = np.nanmean(train[self.fname_mean])
        self.scale_log_same_mean = np.nanstd(train[self.fname_mean])

        self.bias_log_same_count = np.nanmean(train[self.fname_count])
        self.scale_log_same_count = np.nanstd(train[self.fname_count])

        # normalize and fill na
        self.normalize(train)

    def transform(self, test):

        test_merged = pd.merge(test, self.means, how='left')

        test_same_name_count = test_merged['len']
        test_same_name_mean = test_merged['sum'] / test_merged['len']

        test_same_name_count.fillna(0.0, inplace=True)

        test[self.fname_mean] = test_same_name_mean
        test[self.fname_count] = np.log1p(test_same_name_count)

        # normalize and fill na
        self.normalize(test)


# ------------------------------
# global functions
# ------------------------------

def calc_valid_score(valid_predict, valid_prices):
    diff = np.log1p(valid_predict) - np.log1p(valid_prices)
    sums = np.mean(diff * diff)
    score = np.sqrt(sums)
    return score


def common_preprocess(df_data):

    df_data = pd.DataFrame(df_data)

    # fill na
    df_data['name'].fillna('None', inplace=True)

    df_data['item_condition_id'] = pd.to_numeric(df_data['item_condition_id'], errors='coerce')
    df_data['item_condition_id'] = df_data['item_condition_id'].fillna(1).astype(np.int8)

    df_data['category_name'].fillna(value='category_NaN', inplace=True)
    df_data['brand_name'].fillna(value='unknown', inplace=True)

    df_data['shipping'] = pd.to_numeric(df_data['shipping'], errors='coerce')
    df_data['shipping'] = df_data['shipping'].fillna(0).astype(np.int8)

    df_data['item_description'].fillna(value='None', inplace=True)
    df_data['item_description'].replace(value='nodesc', to_replace='No description yet', inplace=True)

    def replace_text(txt):
        txt = txt.lower().replace('16 gb', '16gb').replace('32 gb', '32gb').replace('64 gb', '64gb').\
            replace('128 gb', '128gb').replace('256 gb', '256gb')
        return txt

    df_data['name'] = [replace_text(s) for s in df_data['name'].tolist()]
    df_data['item_description'] = [replace_text(s) for s in df_data['item_description'].tolist()]


# ------------------------------
# main part
# ------------------------------

def main_single(train, valid, extractors, ModelClass, model_class_name, tf_args=None):

    model_abbreviation = model_class_name[0]

    def info_with_abbreviation(info):
        logger.info('[{}] {}'.format(model_abbreviation, info))

    print('-' * 30)
    info_with_abbreviation('==> start {} part'.format(model_class_name))
    tic = time.time()

    info_with_abbreviation('train shape: {}'.format(train.shape))
    info_with_abbreviation('valid shape: {}'.format(valid.shape))

    if tf_args is not None:
        model = ModelClass(*tf_args)
    else:
        model = ModelClass()

    # train
    info_with_abbreviation('fit {} model......'.format(model_class_name))
    model.fit(train)

    if not IS_KERNEL:
        train_predict = model.predict(train)
        train_prices = train['price'].as_matrix()
        train_score = calc_valid_score(train_predict, train_prices)
        info_with_abbreviation('train_score ({}): {:.3f}'.format(model_class_name, train_score))

    # valid
    valid_predict = model.predict(valid)
    valid_prices = valid['price'].as_matrix()
    valid_score = calc_valid_score(valid_predict, valid_prices)
    info_with_abbreviation('valid_score ({}): {:.3f}'.format(model_class_name, valid_score))

    # test (chunk predict)
    test_reader = pd.read_table(PATH_TEST, chunksize=20000)
    test_predict_all = np.array([], dtype=np.float32)

    info_with_abbreviation('predict for test')

    for count, test in prog(enumerate(test_reader)):

        test = pd.DataFrame(test)
        test.reset_index(inplace=True)
        common_preprocess(test)
        [extractor.transform(test) for extractor in extractors]

        test_predict_sub = model.predict(test)
        test_predict_all = np.append(test_predict_all, test_predict_sub)

    # show elasped time
    time_chainer = (time.time() - tic) / 60.0
    info_with_abbreviation('elapsed time ({}) : {:.1f} [min]'.format(model_class_name, time_chainer))

    return test_predict_all, valid_predict


def main_ridge(train, valid, extractors):

    print('-' * 30)
    logger.info('[r] ==> start ridge + fm part')
    tic = time.time()

    logger.info('[r] train shape: {}'.format(train.shape))
    logger.info('[r] valid shape: {}'.format(valid.shape))

    def average_ridge_and_fm(ridge, fm):
        return np.expm1(0.6 * np.log1p(ridge) + 0.4 * np.log1p(fm))

    model = MercariRidge()

    # ------------------------------
    # train

    model.fit(train)

    time_ridge_fm_train = (time.time() - tic) / 60.0
    logger.info('[r] elapsed time (training ridge + fm) : {0:.1f} [min]'.format(time_ridge_fm_train))

    if not IS_KERNEL:

        y_train = train['price'].as_matrix()

        logger.info('[r] predict train (ridge + fm)')
        preds_train_ridge, preds_train_fm = model.predict(train)

        # ridge
        train_score_ridge = calc_valid_score(preds_train_ridge, y_train)
        logger.info('[r] train score (ridge): {0:.3f}'.format(train_score_ridge))

        # fm
        train_score_fm = calc_valid_score(preds_train_fm, y_train)
        logger.info('[r] train score (fm): {0:.3f}'.format(train_score_fm))

        # average
        preds_train = average_ridge_and_fm(preds_train_ridge, preds_train_fm)
        train_score = calc_valid_score(preds_train, y_train)
        logger.info('[r] train score (ridge + fm): {0:.3f}'.format(train_score))

    # ------------------------------
    # valid

    y_valid = valid['price'].as_matrix()

    logger.info('[r] predict valid (ridge + fm)')
    preds_valid_ridge, preds_valid_fm = model.predict(valid)

    # ridge
    valid_score_ridge = calc_valid_score(preds_valid_ridge, y_valid)
    logger.info('[r] valid score (ridge): {0:.3f}'.format(valid_score_ridge))

    # fm
    valid_score_fm = calc_valid_score(preds_valid_fm, y_valid)
    logger.info('[r] valid score (fm): {0:.3f}'.format(valid_score_fm))

    # average result
    valid_predict = average_ridge_and_fm(preds_valid_ridge, preds_valid_fm)
    valid_score = calc_valid_score(valid_predict, y_valid)
    logger.info('[r] valid score (ridge + fm): {0:.3f}'.format(valid_score))

    # ------------------------------
    # test (chunk predict)

    test_reader = pd.read_table(PATH_TEST, chunksize=20000)
    test_predict_all = np.array([], dtype=np.float32)

    logger.info('[r] predict for test')

    for count, test in enumerate(prog(test_reader)):

        test = pd.DataFrame(test)
        test.reset_index(inplace=True)
        common_preprocess(test)
        [extractor.transform(test) for extractor in extractors]

        test_predict_sub_ridge, test_predict_sub_fm = model.predict(test)
        test_predict_sub = average_ridge_and_fm(test_predict_sub_ridge, test_predict_sub_fm)
        test_predict_all = np.append(test_predict_all, test_predict_sub)

    time_ridge_fm = (time.time() - tic) / 60.0
    logger.info('[r] elapsed time (ridge + fm) : {0:.1f} [min]'.format(time_ridge_fm))

    return test_predict_all, valid_predict


def main():

    tic = time.time()

    # load data
    logger.info('==> Load data (train.tsv and test.tsv)')
    train_and_valid = pd.read_table(PATH_TRAIN)

    # split train and valid
    logger.info('==> Split train and valid')
    train, valid = train_test_split(train_and_valid, test_size=0.001, random_state=SEED)
    train.reset_index(inplace=True)
    valid.reset_index(inplace=True)

    train = pd.DataFrame(train)
    valid = pd.DataFrame(valid)

    # execute preprocess (mainly fill NA)
    common_preprocess(train)
    common_preprocess(valid)

    # get same name features
    logger.info('==> Generate same-name features')
    same_name_extractor = SameValueFeatures('name')
    same_brand_extractor = SameValueFeatures('brand_name')
    same_category_extractor = SameValueFeatures('category_name')

    extractors = [same_name_extractor, same_brand_extractor, same_category_extractor]

    [extractor.fit_transform(train) for extractor in extractors]
    [extractor.transform(valid) for extractor in extractors]

    logger.info('==> make word list and load pre-train vector for TF')
    tf_args = MercariTF_init().fit(train)

    # parallel predict by chainer and ridge/fastFM
    result_chainer = pool.apply_async(main_single, (train, valid, extractors, MercariChainer, 'chainer'))
    result_ridge = pool.apply_async(main_ridge, (train, valid, extractors))

    test_predict_ch, valid_predict_ch = result_chainer.get()

    # predict by tensorflow
    test_predict_tf, valid_predict_tf = main_single(train, valid, extractors, MercariTF, 'tensorflow', tf_args=tf_args)

    test_predict_ridge, valid_predict_ridge = result_ridge.get()

    # averaging
    def take_predict_ave(predict_ch, predict_tf, predict_ridge):
        average_log1p = 0.25 * np.log1p(predict_ch) + 0.3 * np.log1p(predict_tf) + 0.45 * np.log1p(predict_ridge)
        return np.expm1(average_log1p)

    test_predict_ave = take_predict_ave(test_predict_ch, test_predict_tf, test_predict_ridge)
    valid_predict_ave = take_predict_ave(valid_predict_ch, valid_predict_tf, valid_predict_ridge)

    # calc valid score
    valid_score = calc_valid_score(valid_predict_ave, valid['price'])
    logger.info('valid score (ALL): {0:.3f}'.format(valid_score))

    # make submission
    test = pd.read_table(PATH_TEST, usecols=['test_id'])
    submission = pd.DataFrame(test['test_id'], columns=['test_id'])
    submission['price'] = test_predict_ave
    submission.to_csv('submission.csv'.format(valid_score), index=False)

    time_all = (time.time() - tic) / 60.0
    logger.info('elapsed time (ALL) : {0:.1f} [min]'.format(time_all))

if __name__ == '__main__':

    pool = Pool(2)
    main()

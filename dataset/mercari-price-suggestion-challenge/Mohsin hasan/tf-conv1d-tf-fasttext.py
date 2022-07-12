'''
Based on
https://www.kaggle.com/knowledgegrappler/a-simple-nn-solution-with-keras-0-48611-pl
https://www.kaggle.com/lscoelho/tensorflow-starter-conv1d-emb-0-43839-lb-v08?scriptVersionId=2089227

'''
import pyximport
pyximport.install()
import numpy as np


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
os.environ['PYTHONHASHSEED'] = '10000'
np.random.seed(10001)
import random
random.seed(10002)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=6, inter_op_parallelism_threads=5)
from keras import backend
tf.set_random_seed(10003)
backend.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))
import re
import time

import gc
import pandas as pd


from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cross_validation import train_test_split

np.random.seed(1525)

t_start = time.time()

from nltk.stem.porter import PorterStemmer
from fastcache import clru_cache as lru_cache
stemmer = PorterStemmer()
from collections import Counter

from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
import tensorflow as tf
cache_size = 8000  # 1024,6000，12000，8000

@lru_cache(cache_size)
def stem(s):
    return stemmer.stem(s)

whitespace = re.compile(r'\s+')
non_letter = re.compile(r'\W+')

def tokenize(text):
    text = text.lower()
    text = non_letter.sub(' ', text)

    tokens = []

    for t in text.split():
        # t = stem(t)
        tokens.append(t)

    return tokens
develop=False
class Tokenizer:
    def __init__(self, min_df=10, tokenizer=str.split):
        self.min_df = min_df
        self.tokenizer = tokenizer
        self.doc_freq = None
        self.vocab = None
        self.vocab_idx = None
        self.max_len = None

    def fit_transform(self, texts):
        tokenized = []
        doc_freq = Counter()
        n = len(texts)

        for text in texts:
            sentence = self.tokenizer(text)
            tokenized.append(sentence)
            doc_freq.update(set(sentence))

        vocab = sorted([t for (t, c) in doc_freq.items() if c >= self.min_df])
        vocab_idx = {t: (i + 1) for (i, t) in enumerate(vocab)}
        doc_freq = [doc_freq[t] for t in vocab]

        self.doc_freq = doc_freq
        self.vocab = vocab
        self.vocab_idx = vocab_idx

        max_len = 0
        result_list = []
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

if __name__ == '__main__':
    import re

    from collections import Counter

    import tensorflow as tf
    import pandas as pd
    import numpy as np

    from nltk.stem.porter import PorterStemmer
    from fastcache import clru_cache as lru_cache

    from sklearn.model_selection import ShuffleSplit
    from sklearn import metrics

    t_start = time.time()

    stemmer = PorterStemmer()


    def rmse(y_true, y_pred):
        return np.sqrt(metrics.mean_squared_error(y_true, y_pred))


    cache_size = 8000


    @lru_cache(cache_size)
    def stem(s):
        return stemmer.stem(s)


    whitespace = re.compile(r'\s+')
    non_letter = re.compile(r'\W+')


    def tokenize(text):
        text = text.lower()
        text = non_letter.sub(' ', text)

        tokens = []

        for t in text.split():
            # t = stem(t)
            tokens.append(t)

        return tokens


    class Tokenizer:
        def __init__(self, min_df=10, tokenizer=str.split):
            self.min_df = min_df
            self.tokenizer = tokenizer
            self.doc_freq = None
            self.vocab = None
            self.vocab_idx = None
            self.max_len = None

        def fit_transform(self, texts):
            tokenized = []
            doc_freq = Counter()
            n = len(texts)

            for text in texts:
                sentence = self.tokenizer(text)
                tokenized.append(sentence)
                doc_freq.update(set(sentence))

            vocab = sorted([t for (t, c) in doc_freq.items() if c >= self.min_df])
            vocab_idx = {t: (i + 1) for (i, t) in enumerate(vocab)}
            doc_freq = [doc_freq[t] for t in vocab]

            self.doc_freq = doc_freq
            self.vocab = vocab
            self.vocab_idx = vocab_idx

            max_len = 0
            result_list = []
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


    print('reading train data...')
    df_train = pd.read_csv('../input/train.tsv', sep='\t')
    df_train = df_train[df_train.price != 0].reset_index(drop=True)

    price = df_train.pop('price')
    y = np.log1p(price.values).reshape(-1, 1)
    mean = y.mean()
    std = y.std()
    ynorm = (y - mean) / std
    ynorm = ynorm.reshape(-1, 1)

    df_train.name.fillna('unkname', inplace=True)
    df_train.category_name.fillna('unk_cat', inplace=True)
    df_train.brand_name.fillna('unk_brand', inplace=True)
    df_train.item_description.fillna('nodesc', inplace=True)

    print('processing category...')


    def paths(tokens):
        all_paths = ['/'.join(tokens[0:(i + 1)]) for i in range(len(tokens))]
        return ' '.join(all_paths)


    @lru_cache(cache_size)
    def cat_process(cat):
        cat = cat.lower()
        cat = whitespace.sub('', cat)
        split = cat.split('/')
        return paths(split)


    df_train.category_name = df_train.category_name.apply(cat_process)

    cat_tok = Tokenizer(min_df=10)
    X_cat = cat_tok.fit_transform(df_train.category_name)
    cat_voc_size = cat_tok.vocabulary_size()
    print(cat_voc_size)

    print('processing title...')

    name_tok = Tokenizer(min_df=5, tokenizer=tokenize)
    X_name = name_tok.fit_transform(df_train.name)
    name_voc_size = name_tok.vocabulary_size()
    print(name_voc_size)

    print('processing description...')

    desc_num_col = 40
    desc_tok = Tokenizer(min_df=10, tokenizer=tokenize)
    X_desc = desc_tok.fit_transform(df_train.item_description)
    X_desc = X_desc[:, :desc_num_col]
    desc_voc_size = desc_tok.vocabulary_size()
    print(desc_voc_size)

    print('processing brand...')

    df_train.brand_name = df_train.brand_name.str.lower()
    df_train.brand_name = df_train.brand_name.str.replace(' ', '_')

    brand_cnt = Counter(df_train.brand_name[df_train.brand_name != 'unk_brand'])
    brands = sorted(b for (b, c) in brand_cnt.items() if c >= 5)
    brands_idx = {b: (i + 1) for (i, b) in enumerate(brands)}

    X_brand = df_train.brand_name.apply(lambda b: brands_idx.get(b, 0))
    X_brand = X_brand.values.reshape(-1, 1)
    brand_voc_size = len(brands) + 1
    print(brand_voc_size)

    print('processing other features...')

    X_item_cond = (df_train.item_condition_id - 1).astype('uint8').values.reshape(-1, 1)
    X_shipping = df_train.shipping.astype('float32').values.reshape(-1, 1)

    print('defining the model...')


    def prepare_batches(seq, step):
        n = len(seq)
        res = []
        for i in range(0, n, step):
            res.append(seq[i:i + step])
        return res


    def conv1d(inputs, num_filters, filter_size, padding='same'):
        he_std = np.sqrt(2 / (filter_size * num_filters))
        out = tf.layers.conv1d(
            inputs=inputs, filters=num_filters, padding=padding,
            kernel_size=filter_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(stddev=he_std))
        return out


    def dense(X, size, reg=0.0, activation=None):
        he_std = np.sqrt(2 / int(X.shape[1]))
        out = tf.layers.dense(X, units=size, activation=activation,
                              kernel_initializer=tf.random_normal_initializer(stddev=he_std),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))
        return out


    def embed(inputs, size, dim):
        std = np.sqrt(2 / dim)
        emb = tf.Variable(tf.random_uniform([size, dim], -std, std))
        lookup = tf.nn.embedding_lookup(emb, inputs)
        return lookup


    name_embeddings_dim = 64
    name_seq_len = X_name.shape[1]
    desc_embeddings_dim = 64
    desc_seq_len = X_desc.shape[1]

    brand_embeddings_dim = 32

    cat_embeddings_dim = 32
    cat_seq_len = X_cat.shape[1]

    graph = tf.Graph()
    graph.seed = 1

    with graph.as_default():
        place_name = tf.placeholder(tf.int32, shape=(None, name_seq_len))
        place_desc = tf.placeholder(tf.int32, shape=(None, desc_seq_len))
        place_brand = tf.placeholder(tf.int32, shape=(None, 1))
        place_cat = tf.placeholder(tf.int32, shape=(None, cat_seq_len))
        place_ship = tf.placeholder(tf.float32, shape=(None, 1))
        place_cond = tf.placeholder(tf.uint8, shape=(None, 1))

        place_y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

        place_lr = tf.placeholder(tf.float32, shape=(), )

        name = embed(place_name, name_voc_size, name_embeddings_dim)
        desc = embed(place_desc, desc_voc_size, desc_embeddings_dim)
        brand = embed(place_brand, brand_voc_size, brand_embeddings_dim)
        cat = embed(place_cat, cat_voc_size, cat_embeddings_dim)

        name = conv1d(name, num_filters=20, filter_size=3)
        name = tf.layers.dropout(name, rate=0.01)
        name = tf.layers.average_pooling1d(name, pool_size=name_seq_len, strides=1, padding='valid')
        name = tf.contrib.layers.flatten(name)
        print(name.shape)

        desc = conv1d(desc, num_filters=20, filter_size=3)
        desc = tf.layers.dropout(desc, rate=0.2)
        desc = tf.layers.average_pooling1d(desc, pool_size=desc_seq_len, strides=1, padding='valid')

        desc = tf.contrib.layers.flatten(desc)
        print(desc.shape)

        brand = tf.contrib.layers.flatten(brand)
        print(brand.shape)

        cat = tf.layers.average_pooling1d(cat, pool_size=cat_seq_len, strides=1, padding='valid')
        cat = tf.contrib.layers.flatten(cat)
        print(cat.shape)

        ship = place_ship
        print(ship.shape)

        cond = tf.one_hot(place_cond, 5)
        cond = tf.contrib.layers.flatten(cond)
        print(cond.shape)

        out = tf.concat([name, desc, brand, cat, ship, cond], axis=1)
        print('concatenated dim:', out.shape)
        # out = tf.contrib.layers.batch_norm(out, decay=0.9)
        out = dense(out, 300, activation=tf.nn.relu)
        out = tf.layers.dropout(out, rate=0.0)
        # out = dense(out, 64, activation=tf.nn.relu)
        # out = tf.layers.dropout(out, rate=0.03)
        # out = tf.contrib.layers.batch_norm(out, decay=0.9)
        out = dense(out, 1)

        loss = tf.losses.mean_squared_error(place_y, out)
        rmse = tf.sqrt(loss)
        opt = tf.train.AdamOptimizer(learning_rate=place_lr)
        train_step = opt.minimize(loss)

        init = tf.global_variables_initializer()

    session = tf.Session(config=None, graph=graph)
    session.run(init)

    print('training the model...')

    train_idx, val_idx = list(ShuffleSplit(1, test_size=0.02, random_state=1).split(X_name))[0]
    lr_init = 0.004
    lr_decay = 0.0014
    lr = lr_init
    for i in range(3):
        t0 = time.time()
        np.random.seed(i)
        np.random.shuffle(train_idx)
        batches = prepare_batches(train_idx, 1000)  # 500

        if i == 1:
            lr = 0.004
        elif i == 2:
            lr = 0.001
        elif i == 3:
            lr = 0.00001
        # lr = lr_init - lr_decay*i
        print(lr)
        for j, idx in enumerate(batches):
            # for j, idx in tqdm(enumerate(batches)):
            feed_dict = {
                place_name: X_name[idx],
                place_desc: X_desc[idx],
                place_brand: X_brand[idx],
                place_cat: X_cat[idx],
                place_cond: X_item_cond[idx],
                place_ship: X_shipping[idx],
                place_y: ynorm[idx],
                place_lr: lr,
            }
            session.run(train_step, feed_dict=feed_dict)

        took = time.time() - t0
        print('Training epoch %d took %.3fs' % (i, took))
        val_batches = prepare_batches(val_idx, 80000)  # 5000
        y_pred = np.zeros(len(X_name))
        for idx in val_batches:
            feed_dict = {
                place_name: X_name[idx],
                place_desc: X_desc[idx],
                place_brand: X_brand[idx],
                place_cat: X_cat[idx],
                place_cond: X_item_cond[idx],
                place_ship: X_shipping[idx],
            }
            batch_pred = session.run(out, feed_dict=feed_dict)
            y_pred[idx] = batch_pred[:, 0]
        y_pred_val = y_pred[val_idx] * std + mean
        y_true_val = ynorm[val_idx][:, 0] + mean
        print("Validation rmse is ", np.sqrt(metrics.mean_squared_error(y_true_val, y_pred_val)))

    print('reading the test data...')

    df_test = pd.read_csv('../input/test.tsv', sep='\t')

    df_test.name.fillna('unkname', inplace=True)
    df_test.category_name.fillna('unk_cat', inplace=True)
    df_test.brand_name.fillna('unk_brand', inplace=True)
    df_test.item_description.fillna('nodesc', inplace=True)

    df_test.category_name = df_test.category_name.apply(cat_process)
    df_test.brand_name = df_test.brand_name.str.lower()
    df_test.brand_name = df_test.brand_name.str.replace(' ', '_')

    X_cat_test = cat_tok.transform(df_test.category_name)
    X_name_test = name_tok.transform(df_test.name)

    X_desc_test = desc_tok.transform(df_test.item_description)
    X_desc_test = X_desc_test[:, :desc_num_col]

    X_item_cond_test = (df_test.item_condition_id - 1).astype('uint8').values.reshape(-1, 1)
    X_shipping_test = df_test.shipping.astype('float32').values.reshape(-1, 1)

    X_brand_test = df_test.brand_name.apply(lambda b: brands_idx.get(b, 0))
    X_brand_test = X_brand_test.values.reshape(-1, 1)

    print('applying the model to test...')

    n_test = len(df_test)
    y_pred = np.zeros(n_test)

    test_idx = np.arange(n_test)
    batches = prepare_batches(test_idx, 80000)

    for idx in batches:
        feed_dict = {
            place_name: X_name_test[idx],
            place_desc: X_desc_test[idx],
            place_brand: X_brand_test[idx],
            place_cat: X_cat_test[idx],
            place_cond: X_item_cond_test[idx],
            place_ship: X_shipping_test[idx],
        }
        batch_pred = session.run(out, feed_dict=feed_dict)
        y_pred[idx] = batch_pred[:, 0]

    y_pred = y_pred * std + mean
    # y_pred = np.expm1(y_pred)


    print('writing the results...')

    test_preds =  y_pred

    del y_pred
    # -----------tf fasttext again--------------------------------------------
    @lru_cache(cache_size)
    def stem(s):
        return stemmer.stem(s)


    whitespace = re.compile(r'\s+')
    non_letter = re.compile(r'\W+')


    def tokenize(text):
        text = text.lower()
        text = non_letter.sub(' ', text)

        tokens = []

        for t in text.split():
            t = stem(t)
            tokens.append(t)

        return tokens


    class Tokenizer:
        def __init__(self, min_df=10, tokenizer=str.split):
            self.min_df = min_df
            self.tokenizer = tokenizer
            self.doc_freq = None
            self.vocab = None
            self.vocab_idx = None
            self.max_len = None

        def fit_transform(self, texts):
            tokenized = []
            doc_freq = Counter()
            n = len(texts)

            for text in texts:
                sentence = self.tokenizer(text)
                tokenized.append(sentence)
                doc_freq.update(set(sentence))

            vocab = sorted([t for (t, c) in doc_freq.items() if c >= self.min_df])
            vocab_idx = {t: (i + 1) for (i, t) in enumerate(vocab)}
            doc_freq = [doc_freq[t] for t in vocab]

            self.doc_freq = doc_freq
            self.vocab = vocab
            self.vocab_idx = vocab_idx

            max_len = 0
            result_list = []
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


    print('reading train data...')
    df_train = pd.read_csv('../input/train.tsv', sep='\t')
    df_train = df_train[df_train.price != 0].reset_index(drop=True)

    price = df_train.pop('price')
    y = np.log1p(price.values)
    mean = y.mean()
    std = y.std()
    y = (y - mean) / std
    y = y.reshape(-1, 1)

    df_train.name.fillna('unkname', inplace=True)
    df_train.category_name.fillna('unk_cat', inplace=True)
    df_train.brand_name.fillna('unk_brand', inplace=True)
    df_train.item_description.fillna('nodesc', inplace=True)

    print('processing category...')


    def paths(tokens):
        all_paths = ['/'.join(tokens[0:(i + 1)]) for i in range(len(tokens))]
        return ' '.join(all_paths)


    @lru_cache(cache_size)
    def cat_process(cat):
        cat = cat.lower()
        cat = whitespace.sub('', cat)
        split = cat.split('/')
        return paths(split)


    df_train.category_name = df_train.category_name.apply(cat_process)
    # print(u'memory：', psutil.Process(os.getpid()).memory_info().rss)
    # print(u'cpu core：', psutil.cpu_count())
    cat_tok = Tokenizer(min_df=55)
    X_cat = cat_tok.fit_transform(df_train.category_name)
    cat_voc_size = cat_tok.vocabulary_size()

    print('processing title...')

    name_tok = Tokenizer(min_df=10, tokenizer=tokenize)
    X_name = name_tok.fit_transform(df_train.name)
    name_voc_size = name_tok.vocabulary_size()

    print('processing description...')

    desc_num_col = 54  # v0 40
    desc_tok = Tokenizer(min_df=50, tokenizer=tokenize)#50
    X_desc = desc_tok.fit_transform(df_train.item_description)
    X_desc = X_desc[:, :desc_num_col]
    desc_voc_size = desc_tok.vocabulary_size()

    print('processing brand...')

    df_train.brand_name = df_train.brand_name.str.lower()
    df_train.brand_name = df_train.brand_name.str.replace(' ', '_')

    brand_cnt = Counter(df_train.brand_name[df_train.brand_name != 'unk_brand'])
    brands = sorted(b for (b, c) in brand_cnt.items() if c >= 50)
    brands_idx = {b: (i + 1) for (i, b) in enumerate(brands)}

    X_brand = df_train.brand_name.apply(lambda b: brands_idx.get(b, 0))
    X_brand = X_brand.values.reshape(-1, 1)
    brand_voc_size = len(brands) + 1

    print('processing other features...')

    X_item_cond = (df_train.item_condition_id - 1).astype('uint8').values.reshape(-1, 1)
    X_shipping = df_train.shipping.astype('float32').values.reshape(-1, 1)


    # print('defining the model...')
    # print(u'memory：', psutil.Process(os.getpid()).memory_info().rss)
    # print(u'cpu core：', psutil.cpu_count())

    def prepare_batches(seq, step):
        n = len(seq)
        res = []
        for i in range(0, n, step):
            res.append(seq[i:i + step])
        return res


    def conv1d(inputs, num_filters, filter_size, padding='same'):
        #     print("filter_size=",filter_size)
        #     print("num_filters=",num_filters)
        he_std = np.sqrt(2 / (filter_size * num_filters))
        print("he_std=", he_std)
        out = tf.layers.conv1d(
            inputs=inputs, filters=num_filters, padding=padding,
            kernel_size=filter_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(stddev=he_std))
        return out


    def dense(X, size, reg=0.0, activation=None):
        #     print("X.shape[1]",X.shape[1])
        he_std = np.sqrt(2 / int(X.shape[1]))
        #     print("he_std=",he_std)
        out = tf.layers.dense(X, units=size, activation=activation,
                              kernel_initializer=tf.random_normal_initializer(stddev=he_std),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))
        return out


    def embed(inputs, size, dim):
        std = np.sqrt(2 / dim)
        #     print("std=",std)
        emb = tf.Variable(tf.random_uniform([size, dim], -std, std))
        lookup = tf.nn.embedding_lookup(emb, inputs)
        return lookup


    # https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow
    def prelu(_x):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg


    name_embeddings_dim = 32
    name_seq_len = X_name.shape[1]
    desc_embeddings_dim = 70  # 32,50
    desc_seq_len = X_desc.shape[1]

    brand_embeddings_dim = 4

    cat_embeddings_dim = 16
    cat_seq_len = X_cat.shape[1]

    graph = tf.Graph()
    graph.seed = 1

    with graph.as_default():
        place_name = tf.placeholder(tf.int32, shape=(None, name_seq_len))
        place_desc = tf.placeholder(tf.int32, shape=(None, desc_seq_len))
        place_brand = tf.placeholder(tf.int32, shape=(None, 1))
        place_cat = tf.placeholder(tf.int32, shape=(None, cat_seq_len))
        place_ship = tf.placeholder(tf.float32, shape=(None, 1))
        place_cond = tf.placeholder(tf.uint8, shape=(None, 1))

        place_y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

        place_lr = tf.placeholder(tf.float32, shape=(), )

        name = embed(place_name, name_voc_size, name_embeddings_dim)
        desc = embed(place_desc, desc_voc_size, desc_embeddings_dim)
        brand = embed(place_brand, brand_voc_size, brand_embeddings_dim)
        cat = embed(place_cat, cat_voc_size, cat_embeddings_dim)

        #   name = conv1d(name, num_filters=13, filter_size=3)
        name = tf.layers.dropout(name, rate=0.5)  # 0.5
        #     name=tf.layers.max_pooling1d(name,pool_size=2,strides=1)
        name = tf.reduce_mean(name, axis=-2)
        #     name = tf.contrib.layers.flatten(name)

        # print(name.shape)

        #     text_embedding = tf.reduce_mean(tf.nn.embedding_lookup(
        #             text_embedding_w, text_ids), axis=-2)
        #     desc = conv1d(desc, num_filters=11, filter_size=3)
        desc = tf.layers.dropout(desc, rate=0.5)  # 0.5
        #     desc=tf.layers.max_pooling1d(desc,pool_size=2,strides=1)
        desc = tf.reduce_mean(desc, axis=-2)
        #     desc = tf.contrib.layers.flatten(desc)
        # print(desc.shape)

        brand = tf.contrib.layers.flatten(brand)
        # print(brand.shape)

        cat = tf.layers.average_pooling1d(cat, pool_size=cat_seq_len, strides=1, padding='valid')
        cat = tf.contrib.layers.flatten(cat)
        # print(cat.shape)

        ship = place_ship
        # print(ship.shape)

        cond = tf.one_hot(place_cond, 5)
        cond = tf.contrib.layers.flatten(cond)
        # print(cond.shape)

        out = tf.concat([name, desc, brand, cat, ship, cond], axis=1)
        # print('concatenated dim:', out.shape)

        #     out = dense(out, 200, activation=tf.nn.relu)
        out = dense(out, 200, activation=prelu)
        out = tf.layers.dropout(out, rate=0.25)  # 0.52
        #     out = dense(out, 50, activation=tf.nn.relu)
        out = dense(out, 50, activation=prelu)
        out = tf.layers.dropout(out, rate=0.25)
        out = dense(out, 1)

        loss = tf.losses.mean_squared_error(place_y, out)
        rmse = tf.sqrt(loss)
        opt = tf.train.AdamOptimizer(learning_rate=place_lr)
        train_step = opt.minimize(loss)

        init = tf.global_variables_initializer()

    session = tf.Session(config=None, graph=graph)
    session.run(init)

    print('training the model...')
    import psutil

    info = psutil.virtual_memory()
    if develop:
        print(u'memory：', psutil.Process(os.getpid()).memory_info().rss)
    epoch_num = 5  # 4
    for i in range(epoch_num):  # 5
        t0 = time.time()
        np.random.seed(i)
        train_idx_shuffle = np.arange(X_name.shape[0])
        np.random.shuffle(train_idx_shuffle)
        batches = prepare_batches(train_idx_shuffle, 1000)  # 500

        if i <= 2:
            lr = 0.009
        else:
            lr = 0.001

        for idx in batches:
            feed_dict = {
                place_name: X_name[idx],
                place_desc: X_desc[idx],
                place_brand: X_brand[idx],
                place_cat: X_cat[idx],
                place_cond: X_item_cond[idx],
                place_ship: X_shipping[idx],
                place_y: y[idx],
                place_lr: lr,
            }
            session.run(train_step, feed_dict=feed_dict)

        took = time.time() - t0
        print('epoch %d took %.3fs' % (i, took))

    # del resource
    del df_train,
    print('reading the test data...')

    df_test = pd.read_csv('../input/test.tsv', sep='\t')
    print(u'memory：', psutil.Process(os.getpid()).memory_info().rss)
    print(u'cpu core：', psutil.cpu_count())
    df_test.name.fillna('unkname', inplace=True)
    df_test.category_name.fillna('unk_cat', inplace=True)
    df_test.brand_name.fillna('unk_brand', inplace=True)
    df_test.item_description.fillna('nodesc', inplace=True)

    df_test.category_name = df_test.category_name.apply(cat_process)
    df_test.brand_name = df_test.brand_name.str.lower()
    df_test.brand_name = df_test.brand_name.str.replace(' ', '_')

    X_cat_test = cat_tok.transform(df_test.category_name)
    X_name_test = name_tok.transform(df_test.name)

    X_desc_test = desc_tok.transform(df_test.item_description)
    X_desc_test = X_desc_test[:, :desc_num_col]

    X_item_cond_test = (df_test.item_condition_id - 1).astype('uint8').values.reshape(-1, 1)
    X_shipping_test = df_test.shipping.astype('float32').values.reshape(-1, 1)

    X_brand_test = df_test.brand_name.apply(lambda b: brands_idx.get(b, 0))
    X_brand_test = X_brand_test.values.reshape(-1, 1)

    print('applying the model to test...')

    n_test = len(df_test)
    y_pred = np.zeros(n_test)

    test_idx = np.arange(n_test)
    batches = prepare_batches(test_idx, 80000)  # 5000
    # ------------
    submission = df_test[['test_id']]

    for idx in batches:
        feed_dict = {
            place_name: X_name_test[idx],
            place_desc: X_desc_test[idx],
            place_brand: X_brand_test[idx],
            place_cat: X_cat_test[idx],
            place_cond: X_item_cond_test[idx],
            place_ship: X_shipping_test[idx],
        }
        batch_pred = session.run(out, feed_dict=feed_dict)
        y_pred[idx] = batch_pred[:, 0]

    del df_test, X_cat_test, X_name_test, X_desc_test, X_brand_test, X_item_cond_test, X_shipping_test
    gc.collect()
    y_pred = y_pred * std + mean
    # y_pred = np.expm1(y_pred)
    # y_pred = np.squeeze(y_pred)
    test_preds = 0.58 * test_preds + 0.42 * y_pred
    print("Write out submission")

    submission['price'] = np.expm1(test_preds)
    submission.price = submission.price.clip(3, 2000)
    submission.to_csv("embedding_nn_v2.csv", index=False)
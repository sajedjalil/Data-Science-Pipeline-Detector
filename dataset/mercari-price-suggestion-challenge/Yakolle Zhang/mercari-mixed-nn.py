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

from keras import backend

tf.set_random_seed(10003)
backend.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))

import gc
import os
import string
import warnings
from multiprocessing import current_process, Process, Queue
from time import time

import pandas as pd
import lightgbm as lgb
import numpy as np
from keras.initializers import glorot_normal
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Conv1D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from scipy.sparse import hstack
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings('ignore')


def insample_outsample_split(x, y, train_size=.5, holdout_num=5, holdout_frac=.7, random_state=0, full_holdout=False):
    if isinstance(train_size, float):
        int(train_size * len(y))

    in_ind, out_ind = ShuffleSplit(n_splits=1, train_size=train_size, test_size=None, random_state=random_state).split(
        y).__next__()
    in_x = x[in_ind]
    in_y = y[in_ind]
    out_x = x[out_ind]
    out_y = y[out_ind]

    out_set = []
    for i in range(holdout_num):
        _, h_ind = ShuffleSplit(n_splits=1, train_size=None, test_size=holdout_frac,
                                random_state=random_state + i).split(out_y).__next__()
        h_x = out_x[h_ind]
        h_y = out_y[h_ind]
        out_set.append((h_x, h_y))

    if full_holdout:
        return in_x, in_y, out_set, out_x, out_y
    return in_x, in_y, out_set


# -------------------------------------------get data---------------------------------------
time_records = {}
mode = 1

name_terms = ['bundle', 'for']
desc_terms = ['.', ',', 'and']


def rec_time(title):
    if mode:
        time_records[title] = int(time())


def print_info(title, message=None):
    if mode:
        last_time = time_records.get(title, 0)
        cur_time = int(time())
        if message is None:
            print('【%s】【%ds】【%s】' % (current_process().name, cur_time - last_time, title))
        else:
            print('【%s】【%ds】【%s】【%s】' % (current_process().name, cur_time - last_time, title, message))
            time_records[0] = cur_time


def extract_cats(col):
    na_val = 'Other'
    cats = col.str.split('/')
    cat_len = cats.str.len()
    cat1 = cats.str.get(0)
    cat2 = cats.str.get(1)
    cat2.fillna(na_val, inplace=True)
    cat3 = cats.str.get(2)
    cat3.fillna(na_val, inplace=True)
    cat_entity = cats.str.get(-1)
    cat_n = cat_entity.copy()
    cat_n.loc[cat_len <= 3] = na_val
    return cat_len, cat1, cat2, cat3, cat_entity, cat_n


def embed_target_agg(col, statistical_size=30, k=5, random_state=0):
    if col.shape[0] < statistical_size:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    else:
        means = []
        stds = []
        mins = []
        q1s = []
        medians = []
        q3s = []
        maxes = []
        for _, vind in KFold(n_splits=k, shuffle=True, random_state=random_state).split(col):
            v_col = col.take(vind)
            means.append(np.mean(v_col))
            stds.append(np.std(v_col))
            mins.append(np.min(v_col))
            q1s.append(np.percentile(v_col, 25))
            medians.append(np.median(v_col))
            q3s.append(np.percentile(v_col, 75))
            maxes.append(np.max(v_col))
        return np.round(np.mean(means)), np.round(np.std(means)), np.round(np.mean(stds)), np.round(
            np.std(stds)), np.round(np.mean(mins)), np.round(np.std(mins)), np.round(np.mean(q1s)), np.round(
            np.std(q1s)), np.round(np.mean(medians)), np.round(np.std(medians)), np.round(np.mean(q3s)), np.round(
            np.std(q3s)), np.round(np.mean(maxes)), np.round(np.std(maxes))


def encode_text(col):
    def count_chars(txt):
        _len = 0
        digit_cnt, number_cnt = 0, 0
        lower_cnt, upper_cnt, letter_cnt, word_cnt = 0, 0, 0, 0
        char_cnt, term_cnt = 0, 0
        conj_cnt, blank_cnt, punc_cnt = 0, 0, 0
        sign_cnt, marks_cnt = 0, 0

        flag = 10
        for ch in txt:
            _len += 1
            if ch in string.ascii_lowercase:
                lower_cnt += 1
                letter_cnt += 1
                char_cnt += 1
                if flag:
                    word_cnt += 1
                    if flag > 2:
                        term_cnt += 1
                    flag = 0
            elif ch in string.ascii_uppercase:
                upper_cnt += 1
                letter_cnt += 1
                char_cnt += 1
                if flag:
                    word_cnt += 1
                    if flag > 2:
                        term_cnt += 1
                    flag = 0
            elif ch in string.digits:
                digit_cnt += 1
                char_cnt += 1
                if 1 != flag:
                    number_cnt += 1
                    if flag > 2:
                        term_cnt += 1
                    flag = 1
            elif '_' == ch:
                conj_cnt += 1
                char_cnt += 1
                if flag > 2:
                    term_cnt += 1
                flag = 2
            elif ch in string.whitespace:
                blank_cnt += 1
                flag = 3
            elif ch in string.punctuation:
                punc_cnt += 1
                flag = 4
            else:
                sign_cnt += 1
                if flag != 5:
                    marks_cnt += 1
                    flag = 5

        return (
            _len, digit_cnt, number_cnt, digit_cnt / (1 + number_cnt), lower_cnt, upper_cnt, letter_cnt, word_cnt,
            letter_cnt / (1 + word_cnt), char_cnt, term_cnt, char_cnt / (1 + term_cnt), conj_cnt, blank_cnt,
            punc_cnt, sign_cnt, marks_cnt, sign_cnt / (1 + marks_cnt))

    return np.array(list(col.apply(count_chars)), dtype=np.uint16)


def desc_cnt_thread(tr_desc_col, ts_desc_col, desc_queue):
    col_name = 'item_description'
    rec_time('%s encode_text' % col_name)
    tr_desc_cnts = encode_text(tr_desc_col)
    ts_desc_cnts = encode_text(ts_desc_col)
    print_info('%s encode_text' % col_name)

    desc_queue.put((tr_desc_cnts, ts_desc_cnts))


def desc_vectorize_thread(tr_desc_col, tr_brand_col, desc_queue):
    col_name = 'item_description'
    rec_time('train %s process' % col_name)
    tr_col = tr_desc_col.str.replace(r'\s\s+', ' ')
    rec_time('train %s vectorize' % col_name)
    vr = CountVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', ngram_range=(1, 3), vocabulary=desc_terms, dtype=np.uint8)
    tr_x_desc = vr.fit_transform(tr_brand_col + ' ' + tr_col)
    print_info('train %s vectorize' % col_name, 'desc vocabulary size is %d.' % len(desc_terms))
    print_info('train %s process' % col_name)

    rec_time('test %s process' % col_name)
    ts_col = desc_queue.get()
    rec_time('test %s vectorize' % col_name)
    ts_x_desc = vr.transform(ts_col)
    print_info('test %s vectorize' % col_name)
    print_info('test %s process' % col_name)

    desc_queue.put((tr_x_desc, ts_x_desc))


def name_thread(tr_name_col, tr_brand_col, ts_name_col, ts_brand_col, name_queue):
    col_name = 'name'
    rec_time('train %s process' % col_name)
    tr_col = tr_name_col.str.replace(r'\s\s+', ' ')
    rec_time('train %s vectorize' % col_name)
    vr = CountVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', ngram_range=(1, 3), vocabulary=name_terms, dtype=np.uint8)
    tr_x_name = vr.fit_transform(tr_brand_col + ' ' + tr_col)
    print_info('train %s vectorize' % col_name, 'name vocabulary size is %d.' % len(name_terms))
    rec_time('train %s encode_text' % col_name)
    tr_name_cnts = encode_text(tr_col)
    print_info('train %s encode_text' % col_name)
    print_info('train %s process' % col_name)

    rec_time('test %s process' % col_name)
    ts_col = ts_name_col.str.replace(r'\s\s+', ' ')
    rec_time('test %s vectorize' % col_name)
    ts_x_name = vr.transform(ts_brand_col + ' ' + ts_col)
    print_info('test %s vectorize' % col_name)
    rec_time('test %s encode_text' % col_name)
    ts_name_cnts = encode_text(ts_col)
    print_info('test %s encode_text' % col_name)
    print_info('test %s process' % col_name)

    name_queue.put((tr_x_name, tr_name_cnts, ts_x_name, ts_name_cnts))


def brand_thread(tr_brand_col, ts_brand_col):
    col_name = 'brand_name'
    rec_time('train %s vectorize' % col_name)
    vr = CountVectorizer(token_pattern='.+', min_df=10, binary=True, dtype=np.uint8)
    tr_x_brand = vr.fit_transform(tr_brand_col)
    print_info('train %s vectorize' % col_name)
    rec_time('test %s vectorize' % col_name)
    ts_x_brand = vr.transform(ts_brand_col)
    print_info('test %s vectorize' % col_name)

    return tr_x_brand, ts_x_brand


def cat_extract_thread(tr_cat_col, ts_cat_col):
    rec_time('extract cats')
    tr_cat_len, tr_cat1, tr_cat2, tr_cat3, tr_cat_entity, tr_cat_n = extract_cats(tr_cat_col)
    ts_cat_len, ts_cat1, ts_cat2, ts_cat3, ts_cat_entity, ts_cat_n = extract_cats(ts_cat_col)
    print_info('extract cats')
    rec_time('calc cat cnt')
    cnts = tr_cat_entity.append(ts_cat_entity).value_counts()
    tr_cat_entity_cnt = tr_cat_entity.to_frame().join(cnts, on='category_name', rsuffix='_cnt')[
        'category_name_cnt']
    ts_cat_entity_cnt = ts_cat_entity.to_frame().join(cnts, on='category_name', rsuffix='_cnt')[
        'category_name_cnt']
    print_info('calc cat cnt')

    return (np.vstack([tr_cat_len, tr_cat_entity_cnt]).T, np.vstack([ts_cat_len, ts_cat_entity_cnt]).T), (
        tr_cat1, tr_cat2, tr_cat3, tr_cat_entity, tr_cat_n), (ts_cat1, ts_cat2, ts_cat3, ts_cat_entity, ts_cat_n)


def cat_thread(tr_cat1, tr_cat2, tr_cat3, tr_cat_n, ts_cat1, ts_cat2, ts_cat3, ts_cat_n, cat_queue):
    rec_time('category_name process')
    col_name = 'cat1'
    vr = CountVectorizer(token_pattern='.+', min_df=30, binary=True, dtype=np.uint8)
    rec_time('%s vectorize' % col_name)
    tr_x_cat1 = vr.fit_transform(tr_cat1)
    ts_x_cat1 = vr.transform(ts_cat1)
    print_info('%s vectorize' % col_name)
    col_name = 'cat2'
    vr = CountVectorizer(token_pattern='.+', min_df=30, binary=True, dtype=np.uint8)
    rec_time('%s vectorize' % col_name)
    tr_x_cat2 = vr.fit_transform(tr_cat2)
    ts_x_cat2 = vr.transform(ts_cat2)
    print_info('%s vectorize' % col_name)
    col_name = 'cat3'
    vr = CountVectorizer(token_pattern='.+', min_df=30, binary=True, dtype=np.uint8)
    rec_time('%s vectorize' % col_name)
    tr_x_cat3 = vr.fit_transform(tr_cat3)
    ts_x_cat3 = vr.transform(ts_cat3)
    print_info('%s vectorize' % col_name)
    col_name = 'cat_n'
    vr = CountVectorizer(token_pattern='.+', min_df=30, binary=True, dtype=np.uint8)
    rec_time('%s vectorize' % col_name)
    tr_x_cat_n = vr.fit_transform(tr_cat_n)
    ts_x_cat_n = vr.transform(ts_cat_n)
    print_info('%s vectorize' % col_name)
    rec_time('combine cats')
    tr_x_cats = hstack((tr_x_cat1, tr_x_cat2, tr_x_cat3, tr_x_cat_n))
    ts_x_cats = hstack((ts_x_cat1, ts_x_cat2, ts_x_cat3, ts_x_cat_n))
    print_info('combine cats')
    print_info('category_name process')

    cat_queue.put((tr_x_cats, ts_x_cats))


def embed_thread(train_df, test_df, key_col_name, extra_cols, random_state=5000):
    rec_time('%s embed target' % key_col_name)
    cols = extra_cols + [key_col_name]
    gp_embed = train_df[cols + ['price']].groupby(cols).agg(embed_target_agg, **{'random_state': random_state})
    embed_col_suffix = '_' + key_col_name + '_embed'
    tr_embed = np.array(list(train_df.join(gp_embed, on=cols, rsuffix=embed_col_suffix)['price' + embed_col_suffix]))
    ts_embed = test_df.join(gp_embed, on=cols)['price']
    ts_embed.loc[ts_embed.isnull()] = [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)] * np.sum(ts_embed.isnull())
    ts_embed = np.array(list(ts_embed))
    print_info('%s embed target' % key_col_name)

    return tr_embed, ts_embed


def item_cond_thread(tr_cond_col, ts_cond_col):
    col_name = 'item_condition_id'
    rec_time('%s process' % col_name)
    tr_x_cond = pd.get_dummies(tr_cond_col).values
    ts_cond_col = ts_cond_col.clip(1, 5)
    ts_x_cond = pd.get_dummies(ts_cond_col).values
    print_info('%s process' % col_name)
    return tr_x_cond, ts_x_cond


def ship_thread(tr_ship_col, ts_ship_col):
    col_name = 'shipping'
    rec_time('%s process' % col_name)
    tr_x_ship = pd.get_dummies(tr_ship_col).values
    ts_x_ship = pd.get_dummies(ts_ship_col).values
    print_info('%s process' % col_name)
    return tr_x_ship, ts_x_ship


def read_data(data_dir='../input'):
    def fillna(df):
        df['brand_name'].fillna('missing', inplace=True)
        df['item_description'].fillna('None', inplace=True)
        df['category_name'].fillna('Other', inplace=True)
        df['name'].fillna('Unk.', inplace=True)
        df.fillna(0, inplace=True)

    rec_time('read data')
    train_df = pd.read_csv(os.path.join(data_dir, 'train.tsv'), sep='\t', engine='c')
    train_df['price'] = np.log1p(train_df.price)
    test_df = pd.read_csv(os.path.join(data_dir, 'test.tsv'), sep='\t', engine='c')
    print_info('read data', '%s %s' % (train_df.shape, test_df.shape))

    rec_time('remove item with 0 price')
    train_df = train_df.loc[train_df.price > 1].reset_index(drop=True)
    print_info('remove item with 0 price', train_df.shape)

    rec_time('fillna')
    fillna(train_df)
    fillna(test_df)
    print_info('fillna', '%s %s' % (train_df.shape, test_df.shape))

    return train_df, test_df


def get_lgb_data(train_df, test_df, share_queue):
    col_name = 'item_description'
    rec_time('train %s normalize' % col_name)
    train_df.item_description.replace('No description yet', 'None', inplace=True)
    print_info('train %s normalize' % col_name)

    desc_queue = Queue()
    desc_worker = get_worker(desc_vectorize_thread, name='desc_vectorize_thread',
                             args=(train_df.item_description, train_df.brand_name, desc_queue))
    desc_worker.start()

    other_queue = Queue()
    other_worker = get_worker(name_thread, name='name_thread',
                              args=(train_df['name'], train_df.brand_name, test_df['name'], test_df.brand_name,
                                    other_queue))
    other_worker.start()

    col_name = 'item_description'
    rec_time('test %s normalize' % col_name)
    test_df.item_description.replace('No description yet', 'None', inplace=True)
    desc_queue.put(test_df.brand_name + ' ' + test_df.item_description.str.replace(r'\s\s+', ' '))
    print_info('test %s normalize' % col_name)

    (tr_cat_cnts, ts_cat_cnts), (tr_cat1, tr_cat2, tr_cat3, train_df['cat_entity'], tr_cat_n), (
        ts_cat1, ts_cat2, ts_cat3, test_df['cat_entity'], ts_cat_n) = cat_extract_thread(train_df.category_name,
                                                                                         test_df.category_name)

    tr_x_cat_embeds, ts_x_cat_embeds = embed_thread(train_df, test_df, 'cat_entity', ['item_condition_id', 'shipping'],
                                                    random_state=5000)
    tr_x_brand_embeds, ts_x_brand_embeds = embed_thread(train_df, test_df, 'brand_name',
                                                        ['item_condition_id', 'shipping'], random_state=5001)

    tr_x_cond, ts_x_cond = item_cond_thread(train_df.item_condition_id, test_df.item_condition_id)

    tr_x_name, tr_name_cnts, ts_x_name, ts_name_cnts = other_queue.get()
    other_worker.join()
    other_worker = get_worker(cat_thread, name='cat_thread',
                              args=(tr_cat1, tr_cat2, tr_cat3, tr_cat_n, ts_cat1, ts_cat2, ts_cat3, ts_cat_n,
                                    other_queue))
    other_worker.start()

    col_name = 'brand_name'
    rec_time('%s process' % col_name)
    cnts = train_df[col_name].append(test_df[col_name]).value_counts()
    train_df = train_df.join(cnts, on=col_name, rsuffix='_cnt')
    test_df = test_df.join(cnts, on=col_name, rsuffix='_cnt')
    train_df[col_name + '_len'] = train_df[col_name].str.len()
    test_df[col_name + '_len'] = test_df[col_name].str.len()
    print_info('%s process' % col_name)
    del cnts

    tr_x_ship, ts_x_ship = ship_thread(train_df.shipping, test_df.shipping)

    cols = ['brand_name_cnt', 'brand_name_len']
    rec_time('extract brand numeric feature')
    tr_x_brand_cnts = train_df[cols]
    ts_x_brand_cnts = test_df[cols]
    print_info('extract brand numeric feature', cols)
    tr_x_brand, ts_x_brand = brand_thread(train_df.brand_name, test_df.brand_name)

    share_queue.put((train_df.category_name, test_df.category_name, train_df['name'], test_df['name']))

    tr_x_cats, ts_x_cats = other_queue.get()
    other_worker.join()
    other_worker = get_worker(desc_cnt_thread, name='desc_cnt_thread',
                              args=(train_df.item_description, test_df.item_description, other_queue))
    other_worker.start()

    tr_desc_cnts, ts_desc_cnts = other_queue.get()
    other_queue.close()
    other_worker.join()
    del other_worker, other_queue
    gc.collect()

    tr_x_desc, ts_x_desc = desc_queue.get()
    desc_queue.close()
    desc_worker.join()
    del desc_worker, desc_queue
    gc.collect()

    rec_time('combine features')
    x_cnts = np.hstack([tr_x_brand_embeds, tr_x_brand_cnts, tr_x_cat_embeds, tr_cat_cnts, tr_name_cnts, tr_desc_cnts])
    ts_x_cnts = np.hstack([ts_x_brand_embeds, ts_x_brand_cnts, ts_x_cat_embeds, ts_cat_cnts, ts_name_cnts,
                           ts_desc_cnts])
    scaler = MinMaxScaler()
    scaled_x_cnts = scaler.fit_transform(x_cnts)
    scaled_ts_x_cnts = scaler.transform(ts_x_cnts)
    x = hstack([tr_x_ship, tr_x_cond, tr_x_brand, tr_x_cats, tr_x_name, tr_x_desc])
    ts_x = hstack([ts_x_ship, ts_x_cond, ts_x_brand, ts_x_cats, ts_x_name, ts_x_desc])
    print_info('combine features')

    return x, train_df.price.copy().values, ts_x, x_cnts, ts_x_cnts, scaled_x_cnts, scaled_ts_x_cnts


def get_worker(func, args, name=None):
    worker = Process(target=func, args=args, name=name)
    return worker


def measure_handler(target, pred):
    return metrics.mean_squared_error(target, pred) ** 0.5


def batch_predict(model, data, batch_num=1, pred_params=None):
    p = []
    batch_size = int(data.shape[0] / batch_num) + 1
    for i in range(batch_num):
        x = data[i * batch_size: (i + 1) * batch_size]
        if pred_params is None:
            p = np.append(p, model.predict(x))
        else:
            p = np.append(p, model.predict(x, **pred_params))
    return p


def run_ols(x, y, test_x, x_cnts, ts_x_cnts):
    rec_time('ols combine features')
    x = hstack([x, x_cnts]).tocsr()
    test_x = hstack([test_x, ts_x_cnts]).tocsr()
    print_info('ols combine features')
    del x_cnts, ts_x_cnts
    gc.collect()

    rec_time('ols insample_outsample_split')
    iind, _ = ShuffleSplit(n_splits=1, train_size=0.8, test_size=None, random_state=3).split(y).__next__()
    ix = x[iind]
    iy = y[iind]
    print_info('ols insample_outsample_split', ix.shape)

    rec_time('ols train')
    ols = LinearRegression(copy_X=False)
    ols.fit(ix, iy)
    print_info('ols train')
    del ix, iy, iind
    gc.collect()

    rec_time('ols predict')
    ols_px = ols.predict(x)
    ols_p = ols.predict(test_x)
    print_info('ols predict')

    return ols_px, ols_p, x, test_x


def run_lgb(x, y, rnn_px, ols_px):
    rec_time('lgb train embedding')
    x = hstack([x, rnn_px, ols_px.reshape(-1, 1)]).tocsr()
    print_info('lgb train embedding')
    del ols_px, rnn_px
    gc.collect()

    rec_time('lgb insample_outsample_split')
    iind, _ = ShuffleSplit(n_splits=1, train_size=0.9, test_size=None, random_state=853).split(y).__next__()
    ix = x[iind]
    iy = y[iind]
    print_info('lgb insample_outsample_split', ix.shape)

    rec_time('lgb train valid split')
    tsize = int(0.9 * iy.shape[0])
    tx = ix[:tsize]
    ty = iy[:tsize]
    print_info('lgb train valid split', tx.shape)
    del ix, iy, iind
    gc.collect()

    # params = {'objective': 'huber', 'metric': 'rmse', 'verbose': -1, 'nthread': 4, 'alpha': 3,
    #           'learning_rate': 0.25, 'num_leaves': 32, 'max_depth': 0, 'min_data': 20, 'bagging_fraction': 1.0,
    #           'feature_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0, 'lambda_l2': 0}
    params = {'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'nthread': 5,
              'learning_rate': 0.175, 'num_leaves': 76, 'max_depth': 13, 'bagging_fraction': 0.9, 'bagging_freq': 1,
              'feature_fraction': 0.9}
    rec_time('lgb train')
    lgb_model = lgb.train(params, lgb.Dataset(tx, label=ty), 3000)
    print_info('lgb train')
    del tx, ty
    gc.collect()
    return lgb_model


def get_nn_data(train_df, test_df, rnn_queue):
    y = train_df.price.values
    x = {}
    ts_x = {}
    len_dic = {}

    cols = ['category_name', 'brand_name']
    for col_name in cols:
        rec_time('%s label encode' % col_name)
        ler = LabelEncoder()
        tr_col = train_df[col_name]
        ts_col = test_df[col_name]
        ler.fit(np.hstack([tr_col, ts_col]))
        ind_col_name = col_name + '_ind'
        x[ind_col_name] = ler.transform(tr_col)
        ts_x[ind_col_name] = ler.transform(ts_col)
        len_dic[ind_col_name] = ler.classes_.shape[0]
        print_info('%s label encode' % col_name)
        del tr_col, ts_col, ler

    def to_sequence(tr_column, ts_column):
        column_name = tr_column.name
        rec_time('%s tokenizer' % column_name)
        tkr = Tokenizer()
        # tkr = Tokenizer(filters='"#$%*+;<=>?@[\\]^_`{|}~\t\n')
        # tkr = Tokenizer(filters='"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        # tkr = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        tkr.fit_on_texts(tr_column)
        tr_column = np.array(tkr.texts_to_sequences(tr_column))
        ts_column = np.array(tkr.texts_to_sequences(ts_column))
        print_info('%s tokenizer' % column_name)
        return tr_column, ts_column, len(tkr.word_index) + 1

    def cat_name_sequence(share_queue):
        tr_cat_col, ts_cat_col, tr_name_col, ts_name_col = share_queue.get()
        share_queue.put(to_sequence(tr_cat_col, ts_cat_col))
        share_queue.put(to_sequence(tr_name_col, ts_name_col))

    max_len_dic = {'category_name_seq': 10, 'item_description_seq': 70, 'name_seq': 10}

    def process_sequence(column_name, tr_column, ts_column):
        rec_time('%s pad_sequences' % column_name)
        x[column_name] = pad_sequences(tr_column, maxlen=max_len_dic[column_name], truncating='post', padding='post')
        ts_x[column_name] = pad_sequences(ts_column, maxlen=max_len_dic[column_name], truncating='post', padding='post')
        print_info('%s pad_sequences' % column_name)

    sequence_worker = get_worker(cat_name_sequence, name='cat_name_sequence', args=(rnn_queue,))
    sequence_worker.start()

    tr_x_desc, ts_x_desc, len_dic['item_description_seq'] = to_sequence(train_df.item_description,
                                                                        test_df.item_description)

    tr_x_cat, ts_x_cat, len_dic['category_name_seq'] = rnn_queue.get()
    process_sequence('category_name_seq', tr_x_cat, ts_x_cat)
    del tr_x_cat, ts_x_cat
    gc.collect()

    process_sequence('item_description_seq', tr_x_desc, ts_x_desc)
    del tr_x_desc, ts_x_desc
    gc.collect()

    test_df['item_condition_id'] = test_df.item_condition_id.astype(np.uint8).clip(1, 5)

    cols = ['item_condition_id', 'shipping']
    for col_name in cols:
        rec_time('onehot %s' % col_name)
        x[col_name] = pd.get_dummies(train_df[col_name]).values
        ts_x[col_name] = pd.get_dummies(test_df[col_name]).values
        print_info('onehot %s' % col_name)

    tr_x_name, ts_x_name, len_dic['name_seq'] = rnn_queue.get()
    process_sequence('name_seq', tr_x_name, ts_x_name)
    del tr_x_name, ts_x_name
    gc.collect()

    rnn_queue.put((x, y, ts_x, len_dic))


def run_rnn(x, y, test_x, len_dic):
    rec_time('rnn tv split')
    iind, oind = ShuffleSplit(n_splits=1, train_size=0.8, test_size=None, random_state=85).split(y).__next__()
    oy = y[oind]
    train_size = int(0.99 * iind.shape[0])
    tind = iind[:train_size]
    ty = y[tind]
    vind = iind[train_size:]
    vy = y[vind]
    tx = {}
    vx = {}
    ox = {}
    for col_name, col in x.items():
        tx[col_name] = col[tind]
        vx[col_name] = col[vind]
        ox[col_name] = col[oind]
    print_info('rnn tv split', '%s %s %s' % (ty.shape, vy.shape, oy.shape))
    del iind, tind, vind
    gc.collect()

    def get_rnn_model():
        name_seq = Input(shape=[tx['name_seq'].shape[1]], name='name_seq')
        desc_seq = Input(shape=[tx['item_description_seq'].shape[1]], name='item_description_seq')
        brand = Input(shape=[1], name='brand_name_ind')
        cat = Input(shape=[1], name='category_name_ind')
        cond = Input(shape=[tx['item_condition_id'].shape[1]], name='item_condition_id')
        ship = Input(shape=[tx['shipping'].shape[1]], name='shipping')
        # cnts = Input(shape=[tx['cnts'].shape[1]], name='cnts')

        emb_name_seq = Embedding(len_dic['name_seq'], 20)(name_seq)
        emb_desc_seq = Embedding(len_dic['item_description_seq'], 25)(desc_seq)
        emb_brand = Embedding(len_dic['brand_name_ind'], 10)(brand)
        emb_cat = Embedding(len_dic['category_name_ind'], 10)(cat)

        # rnn_name_seq = GRU(8)(emb_name_seq)
        # rnn_desc_seq = GRU(12)(emb_desc_seq)
        
        cnn_name_seq = Flatten()(Conv1D(16, kernel_size=3, activation='relu')(emb_name_seq))
        cnn_desc_seq = Flatten()(Conv1D(20, kernel_size=3, activation='relu')(emb_desc_seq))

        # cluster = concatenate([Flatten()(emb_brand), Flatten()(emb_cat), cond, ship])
        # cluster = Dropout(0.2)(Dense(64, activation='relu')(cluster))

        # flat = concatenate([rnn_name_seq, rnn_desc_seq, Flatten()(emb_brand),
        #                     Flatten()(emb_cat), cond, ship])
        # flat = concatenate([cnn_name_seq, cnn_desc_seq, Flatten()(emb_name_seq), Flatten()(emb_desc_seq), 
        #                     Flatten()(emb_brand), Flatten()(emb_cat), cond, ship, cnts])
        flat = concatenate([cnn_name_seq, cnn_desc_seq, Flatten()(emb_brand), Flatten()(emb_cat), cond, ship])

        flat = Dense(256, activation='relu', kernel_initializer=glorot_normal(seed=540))(flat)
        flat = Dense(64, activation='relu', kernel_initializer=glorot_normal(seed=541))(flat)
        output = Dense(1, activation='relu')(flat)

        rnn = Model([name_seq, desc_seq, brand, cat, cond, ship], output)
        rnn.compile(loss='mse', optimizer=Adam(lr=lr_init, decay=lr_decay))
        return rnn

    epochs = 2
    batch_size = 512
    lr_init, lr_fin = 0.007, 0.0005
    steps = int(ty.shape[0] / batch_size) * epochs
    lr_decay = (lr_init/lr_fin)**(1/(steps-1)) - 1
    rnn_model = get_rnn_model()

    rec_time('rnn train')
    rnn_model.fit(tx, ty, epochs=epochs, batch_size=batch_size, verbose=2)
    print_info('rnn train')
    del tx, ty
    gc.collect()

    rec_time('rnn validation')
    v_rmsle = measure_handler(vy, rnn_model.predict(vx))
    print_info('rnn validation', 'valid rmsle: %s' % v_rmsle)
    del vx, vy
    gc.collect()

    rec_time('rnn holdout score')
    h_rmsle = measure_handler(oy, rnn_model.predict(ox, batch_size=70000))
    print_info('rnn holdout score', 'holdout rmsle: %s' % h_rmsle)
    del ox, oy
    gc.collect()

    exit(0)

    batch_size = 70000
    rec_time('rnn embedding predict')
    rnn_px = rnn_model.predict(x, batch_size=batch_size)
    print_info('rnn embedding predict')
    del x, y, len_dic
    gc.collect()

    rec_time('rnn predict')
    rnn_p = rnn_model.predict(test_x, batch_size=batch_size)
    print_info('rnn predict')
    del test_x, rnn_model
    gc.collect()

    return rnn_px, rnn_p


def run_cnn(x, y, test_x, len_dic):
    np.random.seed(10001)

    rec_time('cnn tv split')
    iind, _ = ShuffleSplit(n_splits=1, train_size=0.8, test_size=None, random_state=58).split(y).__next__()
    train_size = int(0.99 * iind.shape[0])
    tind = iind[:train_size]
    ty = y[tind]
    vind = iind[train_size:]
    vy = y[vind]
    tx = {}
    vx = {}
    for col_name, col in x.items():
        tx[col_name] = col[tind]
        vx[col_name] = col[vind]
    print_info('cnn tv split', '%s %s' % (ty.shape, vy.shape))
    del iind, tind, vind
    gc.collect()

    def get_cnn_model():
        name_seq = Input(shape=[tx['name_seq'].shape[1]], name='name_seq')
        desc_seq = Input(shape=[tx['item_description_seq'].shape[1]], name='item_description_seq')
        brand = Input(shape=[1], name='brand_name_ind')
        cat = Input(shape=[1], name='category_name_ind')
        cond = Input(shape=[tx['item_condition_id'].shape[1]], name='item_condition_id')
        ship = Input(shape=[tx['shipping'].shape[1]], name='shipping')
        cnts = Input(shape=[tx['cnts'].shape[1]], name='cnts')

        emb_name_seq = Embedding(len_dic['name_seq'], 20)(name_seq)
        emb_desc_seq = Embedding(len_dic['item_description_seq'], 25)(desc_seq)
        emb_brand = Embedding(len_dic['brand_name_ind'], 10)(brand)
        emb_cat = Embedding(len_dic['category_name_ind'], 10)(cat)

        cnn_name_seq = Flatten()(Conv1D(16, kernel_size=3, activation='relu')(emb_name_seq))
        cnn_desc_seq = Flatten()(Conv1D(20, kernel_size=3, activation='relu')(emb_desc_seq))

        cluster = concatenate([Flatten()(emb_brand), Flatten()(emb_cat), cond, ship])
        cluster = Dropout(0.2)(Dense(64, activation='relu')(cluster))

        flat = concatenate([cnn_name_seq, cnn_desc_seq, cluster, Flatten()(emb_brand), Flatten()(emb_cat),
                            cond, ship, cnts])

        flat = Dropout(0.25)(Dense(256, activation='relu')(flat))
        output = Dense(1, activation='relu')(flat)

        cnn = Model([name_seq, desc_seq, brand, cat, cond, ship, cnts], output)
        cnn.compile(loss='mse', optimizer=Adam(lr=2.5e-3))
        return cnn

    epochs = 2
    batch_size = 750
    cnn_model = get_cnn_model()

    rec_time('cnn train')
    cnn_model.fit(tx, ty, epochs=epochs, batch_size=batch_size, verbose=2)
    print_info('cnn train')
    del tx, ty
    gc.collect()

    rec_time('cnn validation')
    v_rmsle = measure_handler(vy, cnn_model.predict(vx))
    print_info('cnn validation', 'valid rmsle: %s' % v_rmsle)
    del vx, vy, x, y, len_dic
    gc.collect()

    rec_time('cnn predict')
    cnn_p = cnn_model.predict(test_x, batch_size=70000)
    print_info('cnn predict')
    del test_x, cnn_model
    gc.collect()

    return cnn_p


def run():
    train_df, test_df = read_data()
    submission = test_df[['test_id']]

    nn_queue = Queue()
    nn_worker = get_worker(func=get_nn_data, args=(train_df, test_df, nn_queue), name='get_nn_data')
    nn_worker.start()

    x, y, test_x, x_cnts, ts_x_cnts, scaled_x_cnts, scaled_ts_x_cnts = get_lgb_data(train_df, test_df, nn_queue)
    del train_df, test_df
    gc.collect()

    nn_x, nn_y, nn_test_x, len_dic = nn_queue.get()
    nn_queue.close()
    nn_worker.join()
    nn_x['cnts'] = scaled_x_cnts
    nn_test_x['cnts'] = scaled_ts_x_cnts
    del nn_worker, nn_queue
    gc.collect()

    rnn_px, rnn_p = run_rnn(nn_x, nn_y, nn_test_x, len_dic)
    cnn_p = run_cnn(nn_x, nn_y, nn_test_x, len_dic)
    del nn_x, nn_y, nn_test_x, len_dic, scaled_x_cnts, scaled_ts_x_cnts
    gc.collect()

    ols_px, ols_p, x, test_x = run_ols(x, y, test_x, x_cnts, ts_x_cnts)
    del x_cnts, ts_x_cnts
    gc.collect()

    lgb_model = run_lgb(x, y, rnn_px, ols_px)
    del x, y, rnn_px, ols_px
    gc.collect()

    rec_time('lgb test embedding')
    test_x = hstack([test_x, rnn_p, ols_p.reshape(-1, 1)]).tocsr()
    print_info('lgb test embedding')
    rec_time('lgb predict')
    lgb_p = batch_predict(lgb_model, test_x, batch_num=10)
    print_info('lgb predict')
    del lgb_model, test_x
    gc.collect()

    rnn_p = rnn_p.reshape(-1)
    cnn_p = cnn_p.reshape(-1)

    rec_time('save')
    submission['price'] = np.expm1(lgb_p)
    submission.to_csv('lgb_ols_rnn_cnn_100_0_0_0.csv', index=False)

    submission['price'] = np.expm1(ols_p)
    submission.to_csv('lgb_ols_rnn_cnn_0_100_0_0.csv', index=False)

    submission['price'] = np.expm1(rnn_p)
    submission.to_csv('lgb_ols_rnn_cnn_0_0_100_0.csv', index=False)

    submission['price'] = np.expm1(cnn_p)
    submission.to_csv('lgb_ols_rnn_cnn_0_0_0_100.csv', index=False)

    submission['price'] = np.expm1(0.87 * lgb_p + 0.01 * ols_p + 0.02 * rnn_p + 0.1 * cnn_p)
    submission.to_csv('lgb_ols_rnn_cnn_87_1_2_10.csv', index=False)

    submission['price'] = np.expm1(0.87 * lgb_p + 0.01 * ols_p + 0.06 * rnn_p + 0.06 * cnn_p)
    submission.to_csv('lgb_ols_rnn_cnn_87_1_6_6.csv', index=False)

    submission['price'] = np.expm1(0.82 * lgb_p + 0.01 * ols_p + 0.02 * rnn_p + 0.15 * cnn_p)
    submission.to_csv('lgb_ols_rnn_cnn_82_1_2_15.csv', index=False)

    submission['price'] = np.expm1(0.82 * lgb_p + 0.01 * ols_p + 0.08 * rnn_p + 0.09 * cnn_p)
    submission.to_csv('lgb_ols_rnn_cnn_82_1_8_9.csv', index=False)

    submission['price'] = np.expm1(0.92 * lgb_p + 0.01 * ols_p + 0.01 * rnn_p + 0.06 * cnn_p)
    submission.to_csv('lgb_ols_rnn_cnn_92_1_1_6.csv', index=False)

    submission['price'] = np.expm1(0.92 * lgb_p + 0.01 * ols_p + 0.03 * rnn_p + 0.04 * cnn_p)
    submission.to_csv('lgb_ols_rnn_cnn_92_1_3_4.csv', index=False)
    print_info('save')


if __name__ == '__main__':
    run()
import pyximport; pyximport.install()
import os
import string
import warnings
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ShuffleSplit

warnings.filterwarnings('ignore')


def insample_outsample_split(x, y, train_size=.5, holdout_num=5, holdout_frac=.7, random_state=0, full_holdout=False):
    if isinstance(train_size, float):
        int(train_size * len(y))

    train_index, h_index = ShuffleSplit(n_splits=1, train_size=train_size, test_size=None,
                                        random_state=random_state).split(y).__next__()
    tx = x[train_index]
    ty = y[train_index]
    h_x = x[h_index]
    h_y = y[h_index]

    h_set = []
    for i in range(holdout_num):
        off_index, v_index = ShuffleSplit(n_splits=1, train_size=None, test_size=holdout_frac,
                                          random_state=random_state + i).split(h_y).__next__()
        vx = h_x[v_index]
        vy = h_y[v_index]
        h_set.append((vx, vy))

    if full_holdout:
        return tx, ty, h_set, h_x, h_y
    return tx, ty, h_set


# -------------------------------------------get data---------------------------------------
last_time = [int(datetime.now().timestamp())]
all_cols = [[]]


def print_info(title, message=None, mode=1):
    if mode:
        cur_time = int(datetime.now().timestamp())
        if message is None:
            print('【%s】【cur_time(%d), take(%d)s】' % (title, cur_time, cur_time - last_time[0]))
        else:
            print('【%s】【cur_time(%d), take(%d)s】【%s】' % (title, cur_time, cur_time - last_time[0], message))
        last_time[0] = cur_time


def get_data(input_dir):
    def fill_na(df):
        df.category_name.fillna('Other', inplace=True)
        df.brand_name.fillna('missing', inplace=True)
        df.item_description.fillna('None', inplace=True)
        df.fillna('0', inplace=True)

    def normalize_info(df):
        df.item_description.replace('No description yet', 'None', inplace=True)
        df['item_description'] = df.item_description.str.replace(r'\s\s+', ' ')
        df['name'] = df.name.str.replace(r'\s\s+', ' ')

    def fill_brand(df):
        def extract_brand(txt):
            brands = ['lularoe', 'jordan', 'lululemon', 'rae dunn', 'funko', "victoria's secret", 'pokemon',
                      'michael kors', 'louis vuitton', 'nintendo', 'ugg australia', 'gucci', 'pandora', 'adidas',
                      'nike', 'disney', 'apple']
            txt = txt.lower()
            the_brand = 'missing'
            for brand in brands:
                if brand in txt:
                    the_brand = brand
                    break
            return the_brand

        df.loc['missing' == df.brand_name, 'brand_name'] = df.loc['missing' == df.brand_name, 'name'].apply(
            extract_brand)

    def extract_cats(df):
        na_val = 'Other'
        df['cats'] = df.category_name.str.split('/')
        df['cat_len'] = df.cats.str.len()
        df['cat1'] = df.cats.str.get(0)
        df['cat2'] = df.cats.str.get(1)
        df['cat3'] = df.cats.str.get(2)
        df['cat_entity'] = df.cats.str.get(-1)
        df['cat_n'] = na_val
        df.loc[df.cat_len > 3, 'cat_n'] = df.cats.str.get(-1)
        df.fillna(na_val, inplace=True)

    def encode_text(df, col_name, col_abbr):
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

        df[col_abbr + '_len'], df[col_abbr + '_digit_cnt'], df[col_abbr + '_number_cnt'], df[
            col_abbr + '_number_size'], df[col_abbr + '_lower_cnt'], df[col_abbr + '_upper_cnt'], df[
            col_abbr + '_letter_cnt'], df[col_abbr + '_word_cnt'], df[col_abbr + '_word_size'], df[
            col_abbr + '_char_cnt'], df[col_abbr + '_term_cnt'], df[col_abbr + '_term_size'], df[
            col_abbr + '_conj_cnt'], df[col_abbr + '_blank_cnt'], df[col_abbr + '_punc_cnt'], df[
            col_abbr + '_sign_cnt'], df[col_abbr + '_marks_cnt'], df[col_abbr + '_marks_size'] = zip(
            *df[col_name].apply(count_chars))

    print_info('begin')
    train_df = pd.read_csv(os.path.join(input_dir, 'train.tsv'), sep='\t', engine='c')
    train_df['item_condition_id'] = train_df['item_condition_id'].astype(str)
    train_df['shipping'] = train_df['shipping'].astype(str)
    test_df = pd.read_csv(os.path.join(input_dir, 'test.tsv'), sep='\t', engine='c')
    test_df['item_condition_id'] = test_df['item_condition_id'].astype(str)
    test_df['shipping'] = test_df['shipping'].astype(str)

    print_info('read data', train_df.shape)
    train_df = train_df.loc[train_df.price > 1].reset_index(drop=True)
    print_info('remove item with 0 price', train_df.shape)

    fill_na(train_df)
    fill_na(test_df)
    print_info('fill_na')

    normalize_info(train_df)
    normalize_info(test_df)
    print_info('normalize_info')

    # fill_brand(train_df)
    # fill_brand(test_df)
    # print_info('fill_brand')

    extract_cats(train_df)
    extract_cats(test_df)
    print_info('extract_cats')

    encode_text(train_df, 'name', 'nm')
    encode_text(test_df, 'name', 'nm')
    encode_text(train_df, 'item_description', 'desc')
    encode_text(test_df, 'item_description', 'desc')
    print_info('encode_text')

    cols = ['cat_entity', 'brand_name']
    for col in cols:
        cnts = train_df[col].append(test_df[col]).value_counts()
        train_df = train_df.join(cnts, on=col, rsuffix='_cnt')
        test_df = test_df.join(cnts, on=col, rsuffix='_cnt')
        print_info('count category and brand', col)

    train_df['brand_name_len'] = train_df.brand_name.str.len()
    test_df['brand_name_len'] = test_df.brand_name.str.len()
    print_info('brand len')

    cols = ['cat_len', 'nm_len', 'nm_digit_cnt', 'nm_number_cnt', 'nm_number_size', 'nm_lower_cnt', 'nm_upper_cnt',
            'nm_letter_cnt', 'nm_word_cnt', 'nm_word_size', 'nm_char_cnt', 'nm_term_cnt', 'nm_term_size', 'nm_conj_cnt',
            'nm_blank_cnt', 'nm_punc_cnt', 'nm_sign_cnt', 'nm_marks_cnt', 'nm_marks_size', 'desc_len', 'desc_digit_cnt',
            'desc_number_cnt', 'desc_number_size', 'desc_lower_cnt', 'desc_upper_cnt', 'desc_letter_cnt',
            'desc_word_cnt', 'desc_word_size', 'desc_char_cnt', 'desc_term_cnt', 'desc_term_size', 'desc_conj_cnt',
            'desc_blank_cnt', 'desc_punc_cnt', 'desc_sign_cnt', 'desc_marks_cnt', 'desc_marks_size', 'cat_entity_cnt',
            'brand_name_cnt', 'brand_name_len']
    x = train_df[cols].values
    ts_x = test_df[cols].values
    print_info('extract numeric feature done', cols)
    all_cols[0] += cols

    vr = CountVectorizer(token_pattern='\d+')
    cols = ['item_condition_id', 'shipping']
    for col in cols:
        x = hstack((x, vr.fit_transform(train_df[col])))
        ts_x = hstack((ts_x, vr.transform(test_df[col])))
        print_info('vectorize', col)
        all_cols[0] += [col + '_' + str(tk) for tk in vr.get_feature_names()]

    vr = CountVectorizer(token_pattern='.+', min_df=3)
    cols = ['cat1', 'cat2', 'cat3', 'cat_n', 'brand_name']
    for col in cols:
        x = hstack((x, vr.fit_transform(train_df[col])))
        ts_x = hstack((ts_x, vr.transform(test_df[col])))
        print_info('vectorize', col)
        all_cols[0] += [col + '_' + str(tk) for tk in vr.get_feature_names()]

    cols = ['name', 'item_description']
    for col in cols:
        train_df[col] = train_df.brand_name + ' ' + train_df[col]
        test_df[col] = test_df.brand_name + ' ' + test_df[col]
        print_info('combine brand with name or desc', col)

    vr = CountVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', ngram_range=(1, 1), min_df=30)
    col = 'name'
    x = hstack((x, vr.fit_transform(train_df[col])))
    ts_x = hstack((ts_x, vr.transform(test_df[col])))
    print_info(col, len(vr.get_feature_names()))
    all_cols[0] += [col + '_' + str(tk) for tk in vr.get_feature_names()]

    vr = CountVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', ngram_range=(2, 2), max_features=1000, min_df=30)
    col = 'name'
    x = hstack((x, vr.fit_transform(train_df[col])))
    ts_x = hstack((ts_x, vr.transform(test_df[col])))
    print_info(col, '2gram')
    all_cols[0] += [col + '_' + str(tk) for tk in vr.get_feature_names()]

    vr = CountVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', ngram_range=(3, 3), max_features=100, min_df=30)
    col = 'name'
    x = hstack((x, vr.fit_transform(train_df[col])))
    ts_x = hstack((ts_x, vr.transform(test_df[col])))
    print_info(col, '3gram')
    all_cols[0] += [col + '_' + str(tk) for tk in vr.get_feature_names()]

    vr = CountVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', ngram_range=(1, 1), min_df=30)
    col = 'item_description'
    print(col)
    x = hstack((x, vr.fit_transform(train_df[col])))
    ts_x = hstack((ts_x, vr.transform(test_df[col])))
    print_info(col, len(vr.get_feature_names()))
    all_cols[0] += [col + '_' + str(tk) for tk in vr.get_feature_names()]

    vr = CountVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', ngram_range=(2, 2), max_features=5000, min_df=30)
    col = 'item_description'
    x = hstack((x, vr.fit_transform(train_df[col])))
    ts_x = hstack((ts_x, vr.transform(test_df[col])))
    print_info(col, '2gram')
    all_cols[0] += [col + '_' + str(tk) for tk in vr.get_feature_names()]

    vr = CountVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', ngram_range=(3, 3), max_features=500, min_df=30)
    col = 'item_description'
    x = hstack((x, vr.fit_transform(train_df[col])))
    ts_x = hstack((ts_x, vr.transform(test_df[col])))
    print_info(col, '3gram')
    all_cols[0] += [col + '_' + str(tk) for tk in vr.get_feature_names()]

    return x.tocsr(), np.log1p(train_df.price.values), ts_x.tocsr(), test_df[['test_id']]


data_dir = '../input'
X, y, test_x, submission = get_data(data_dir)
print_info('tocsr', '%s %s' % (X.shape, test_x.shape))
t_ind, v_ind = ShuffleSplit(n_splits=1, train_size=0.8, test_size=None, random_state=1008).split(y).__next__()
tx = X[t_ind]
ty = y[t_ind]
vx = X[v_ind]
vy = y[v_ind]
print_info('train valid split', '%s %s %s %s' % (tx.shape, ty.shape, vx.shape, vy.shape))


def measure_handler(target, pred):
    return metrics.mean_squared_error(target, pred) ** 0.5


params = {'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'nthread': 4,
          'learning_rate': 0.28, 'num_leaves': 32, 'max_depth': 8, 'min_data': 20}
model = lgb.train(params, lgb.Dataset(tx, label=ty), 5200, valid_sets=[lgb.Dataset(vx, label=vy)],
                  early_stopping_rounds=500, verbose_eval=500)
print_info('train done', 'Best iteration: %d' % model.best_iteration)
# model = lgb.train(params, lgb.Dataset(tx, label=ty), num_boost_round=5180)
# print_info('train done')

print('feature split importance:')
ims = []
feature_importance = model.feature_importance('split')
print(np.max(feature_importance), np.min(feature_importance), np.mean(feature_importance))
for i in range(len(feature_importance)):
    ims.append((all_cols[0][i], feature_importance[i]))
ims = sorted(ims, key=lambda pair: pair[1], reverse=True)
im_out = pd.DataFrame()
im_out['col'] = [col for col, im in ims]
im_out['importance'] = [im for col, im in ims]
print(im_out)
im_out.to_csv('ims.csv', index=False)
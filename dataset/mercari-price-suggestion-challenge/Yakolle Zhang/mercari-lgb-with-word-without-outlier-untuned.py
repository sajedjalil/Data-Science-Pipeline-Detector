import gc
import os
import re
import string
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
def get_data(input_dir):
    def fill_na(df):
        df.fillna('0', inplace=True)
        df.item_description.replace('No description yet', '0', inplace=True)

    def extract_cats(df):
        df['cats'] = df.category_name.str.split('/')
        df['cat_len'] = df.cats.str.len()
        df['cat1'] = df.cats.str.get(0)
        df['cat2'] = df.cats.str.get(1)
        df['cat3'] = df.cats.str.get(2)
        df['cat_n'] = df.cats.str.get(-1)
        df.fillna('0', inplace=True)
        df.drop(['cats'], axis=1, inplace=True)

    def encode_text(df, col, col_abbr):
        df[col_abbr + '_len'] = df[col].str.len()
        df[col_abbr + '_punc_cnt'] = df[col].str.count('[' + re.escape(string.punctuation) + ']')
        df[col_abbr + '_word_cnt'] = df[col].str.count('[' + re.escape(string.whitespace) + ']') + 1
        df[col_abbr + '_word_size'] = (df[col_abbr + '_len'] - df[col_abbr + '_punc_cnt']) / df[col_abbr + '_word_cnt']

    train_df = pd.read_csv(os.path.join(input_dir, 'train.tsv'), sep='\t', engine='c')
    train_df['item_condition_id'] = train_df['item_condition_id'].astype(str)
    train_df['shipping'] = train_df['shipping'].astype(str)
    test_df = pd.read_csv(os.path.join(input_dir, 'test.tsv'), sep='\t', engine='c')
    test_df['item_condition_id'] = test_df['item_condition_id'].astype(str)
    test_df['shipping'] = test_df['shipping'].astype(str)

    fill_na(train_df)
    fill_na(test_df)

    extract_cats(train_df)
    extract_cats(test_df)

    gp = train_df.groupby(['item_condition_id', 'shipping', 'cat_n'])
    stat = gp.count().price.to_frame()
    stat.rename(columns={'price': 'cnt'}, inplace=True)
    stat['mean'] = gp.mean().price
    stat['std'] = gp.std().price
    stat = stat.loc[stat.cnt > 30]

    trf = train_df.join(stat, on=['item_condition_id', 'shipping', 'cat_n'])
    otrf = trf.loc[(trf.price > trf['mean'] + 8 * trf['std']) | (trf.price < trf['mean'] - 8 * trf['std'])]
    trf = trf.drop(otrf.index, axis=0)
    train_df = trf.append(otrf, ignore_index=True)
    trf_len = trf.shape[0]
    print(trf_len, otrf.shape[0])

    encode_text(train_df, 'name', 'name')
    encode_text(test_df, 'name', 'name')
    encode_text(train_df, 'item_description', 'desc')
    encode_text(test_df, 'item_description', 'desc')

    print('encode...')
    cols = ['cat_len', 'name_len', 'name_punc_cnt', 'name_word_cnt', 'name_word_size', 'desc_len', 'desc_punc_cnt',
            'desc_word_cnt', 'desc_word_size']
    print(cols)
    x = train_df[cols].values
    ts_x = test_df[cols].values

    cvr = CountVectorizer(token_pattern='\d+')
    cols = ['item_condition_id', 'shipping', 'cat1', 'cat2', 'cat3', 'cat_n', 'brand_name']
    for col in cols:
        print(col)
        x = hstack((x, cvr.fit_transform(train_df[col])))
        ts_x = hstack((ts_x, cvr.transform(test_df[col])))

    cvr = CountVectorizer(ngram_range=(1, 2), max_features=10000, min_df=30)
    col = 'name'
    print(col)
    x = hstack((x, cvr.fit_transform(train_df[col])))
    ts_x = hstack((ts_x, cvr.transform(test_df[col])))

    tvr = CountVectorizer(ngram_range=(1, 2), max_features=50000, min_df=30)
    col = 'item_description'
    print(col)
    x = hstack((x, tvr.fit_transform(train_df[col]))).tocsr()
    ts_x = hstack((ts_x, tvr.transform(test_df[col]))).tocsr()

    ox = x[trf_len:]
    x = x[:trf_len]
    target = np.log1p(train_df.price.values)

    return x, target[:trf_len], ts_x, test_df[['test_id']], ox, target[trf_len:]


data_dir = '../input'
X, y, test_x, submission, ox, oy = get_data(data_dir)
print(np.mean(y), np.std(y), np.mean(oy), np.std(oy))
print(X.shape, test_x.shape, ox.shape)
train_x, train_y, holdout_data_set = insample_outsample_split(X, y, train_size=0.9, random_state=853)
print(train_x.shape)
del X, y
gc.collect()


def measure_handler(target, pred):
    return metrics.mean_squared_error(target, pred) ** 0.5


print('train begin...')
params = {'objective': 'huber', 'metric': 'rmse', 'verbose': -1, 'nthread': 4, 'alpha': 3,
          'learning_rate': 0.3, 'num_leaves': 32, 'max_depth': 8, 'min_data': 20, 'feature_fraction': 0.6}
model = lgb.train(params, lgb.Dataset(train_x, label=train_y), num_boost_round=5808)
print('train done.')

# valid_scores = []
# for valid_x, valid_y in holdout_data_set:
#     valid_scores.append(measure_handler(valid_y, model.predict(valid_x)))
# print(valid_scores)
# print('holdout_mean=', np.mean(valid_scores), ', holdout_std=', np.std(valid_scores))
# print('----------------------------------------------------------------------------')

p = model.predict(test_x)
submission['price'] = np.expm1(p)
print('save begin...')
submission.to_csv('lgb_with_word_untuned.csv', index=False)
print('save done.')
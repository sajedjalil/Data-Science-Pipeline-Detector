'''
Based on https://www.kaggle.com/apapiu/ridge-script

Optimization:
- Guess from description whether item had pictures
- Guess item quantity from name
'''

import pandas as pd
import numpy as np
import scipy
import math
import re
from multiprocessing import Pool

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, ParameterGrid
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

import lightgbm as lgb

import gc

n_threads = 4


def rmsle(y, y_pred, *args, **kwargs):
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1))
              ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0 / len(y))) ** 0.5

def split_cat(text):
    try: return text.split("/")
    except: return (None, None, None)

def get_qntys(data):
    qnty_matches = []
    qnty_re = [r'(\d+) ?x [^\d]', r'(\d+) ?pairs?']
    for r in qnty_re:
        qnty_matches.append(data.name.str.extract(
            r, flags=re.IGNORECASE, expand=False).dropna().astype(int))
    return pd.concat(qnty_matches).reset_index().drop_duplicates(subset='index', keep='last').set_index('index')


def may_have_pictures(descriptions):
    pic_word_re = [re.compile(r, re.IGNORECASE) for r in [
        r'((see(n)?)|( in| the| my))+ (picture(s)?|photo(s)?)']]
    matches = []
    for desc in descriptions:
        match = 0
        for r in pic_word_re:
            if r.search(desc) is not None:
                match = 1
                continue
        matches.append(match)
    return np.array(matches)

NUM_BRANDS = 3000
NAME_MIN_DF = 10
MAX_FEAT_DESCP = 50000
LOAD_MODEL = False

if __name__ == '__main__':
    print("Reading data")

    df_train = pd.read_csv('../input/train.tsv', sep='\t')
    df_test = pd.read_csv('../input/test.tsv', sep='\t')

    df = pd.concat([df_train, df_test], 0)
    nrow_train = df_train.shape[0]
    y_train = np.log1p(df_train["price"])

    del df_train
    gc.collect()

    df['category_name'], df['subcat_1'], df['subcat_2'] = zip(*df['category_name'].apply(lambda x: split_cat(x)))

    df['category_name'].fillna(value='missing', inplace=True)
    df['subcat_1'].fillna(value='missing', inplace=True)
    df['subcat_2'].fillna(value='missing', inplace=True)
    df['brand_name'].fillna(value='missing', inplace=True)
    df['item_description'].fillna(value='missing', inplace=True)

    print('Getting quantities')
    df['qnty'] = get_qntys(df)
    df.fillna(value={'qnty': 1}, inplace=True)
    df.drop('name', axis=1, inplace=True)
    gc.collect()

    print('Adding information whether item might have containted pictures')
    df['pics'] = pd.Series(may_have_pictures(df.item_description), index=df.index).astype('category')

    pop_brand = df['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    df.loc[~df['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category = df['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    df.loc[~df['category_name'].isin(pop_category), 'category_name'] = 'missing'

    df['category_name'] = df['category_name'].astype('category')
    df['brand_name'] = df['brand_name'].astype('category')
    df['item_condition_id'] = df['item_condition_id'].astype('category')

    print("Count-vectorizing categories")
    count_category = CountVectorizer()
    X_category = count_category.fit_transform(df["category_name"])
    X_category_s1 = count_category.fit_transform(df["subcat_1"])
    X_category_s2 = count_category.fit_transform(df["subcat_2"])

    print("Getting descriptions' TF-IDFs")
    count_descp = TfidfVectorizer(max_features=MAX_FEAT_DESCP,
                                  ngram_range=(1, 3),
                                  stop_words="english")
    X_descp = count_descp.fit_transform(df["item_description"])

    print("Encoding brands")
    vect_brand = CountVectorizer()
    X_brand = vect_brand.fit_transform(df["brand_name"])

    print("Creating dummies")
    X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(df[["item_condition_id", "shipping", "pics"]], sparse=True).values)

    X_qnty = scipy.sparse.csr_matrix(df['qnty']).T

    X = scipy.sparse.hstack((
                            X_dummies,
                            X_descp,
                            X_brand,
                            X_category,
                            X_category_s1,
                            X_category_s2,
                            X_qnty
                            )).tocsr()

    X_test = X[nrow_train:]
    X_train = X[:nrow_train]
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=123, train_size=0.98)

    GRID_SEARCH = False
    param_grid = {
        'lgb_lr': [0.75],
        'lgb_max_bin': [8192, 4096, 16384],
        'lgb_num_leaves': [32, 100, 1000],
        'lgb_max_depth': [3, 8, 16],
        'lgb_num_trees': [750, 2000, 3000]
    }

    if not GRID_SEARCH:
        for k, v in param_grid.items():
            param_grid[k] = [v[0]]

    best_params = None
    best_scores = (999,)
    best_model_gb = None

    print('Non-cross-validated grid search about to evaluate {} param combinations'.format(len(list(ParameterGrid(param_grid)))))

    for p in list(ParameterGrid(param_grid)):
        print('Evaluating param set: {}'.format(p))

        d_train = lgb.Dataset(X_train, label=y_train, max_bin=p['lgb_max_bin'])
        d_valid = lgb.Dataset(X_valid, label=y_valid, max_bin=p['lgb_max_bin'])
        watchlist = [d_train, d_valid]

        params = {
            'learning_rate': p['lgb_lr'], # caution: params dict is modified by lgb
            'application': 'regression',
            'max_depth': p['lgb_max_depth'],
            'num_leaves': p['lgb_num_leaves'],
            'verbosity': -1,
            'metric': 'RMSE',
        }

        print("Fitting boosted trees")
        model_gb = lgb.train(params, train_set=d_train, num_boost_round=p['lgb_num_trees'],
                                    valid_sets=watchlist, early_stopping_rounds=50, verbose_eval=0,
                                    callbacks=[])

        print("Evaluating model")
        preds_gb = np.array(model_gb.predict(X_valid))
        preds = preds_gb
        score = mean_squared_log_error(y_valid, preds) ** 0.5

        if score < best_scores[0]:
            best_scores = (score,)
            best_params = p
            best_model_gb = model_gb

    print('Best score: {}'.format(best_scores[0]))
    print('Best params: {}'.format(best_params))

    model_gb = best_model_gb

    print('Predicting on test set')
    preds_gb = np.array(model_gb.predict(X_test))
    preds = preds_gb
    df_test["price"] = np.expm1(preds)
    print('Saving submission')
    df_test[["test_id", "price"]].to_csv("submission_ridge.csv", index=False)

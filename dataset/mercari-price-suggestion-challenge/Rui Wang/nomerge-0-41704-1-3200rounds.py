import gc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import sys
#Add https://www.kaggle.com/anttip/wordbatch to your kernel Data Sources
sys.path.insert(0, '../input/wordbatch/wordbatch/')
import wordbatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL
from nltk.corpus import stopwords
import re

NUM_BRANDS = 3750
NUM_CATEGORIES = 1200
develop = True

def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    return np.sqrt(np.mean(np.square( y - y_pred )))

def handle_missing_inplace(dataset):
    dataset['shipping'].fillna(0, inplace=True)
    dataset['item_condition_id'].fillna(1, inplace=True)
    dataset['name'].fillna(value='missing', inplace=True)
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='No description yet', inplace=True)
    dataset.loc[pd.isnull(dataset['brand_name']), 'brand_name'] = dataset['name'].apply(lambda x: x.split()[0])

def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['category_name'].isin(pop_category), 'category_name'] = 'missing'

def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')
    dataset['shipping'] = dataset['shipping'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')

stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')

def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])

def best_ratio_finder(Y1, Y2, ratio):
    assert Y1.shape == Y2.shape
    return Y1 * ratio + Y2 * (1.0 - ratio)
# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def main():

    train = pd.read_table('../input/mercari-price-suggestion-challenge/train.tsv', engine='c')
    test = pd.read_table('../input/mercari-price-suggestion-challenge/test.tsv', engine='c')
    print('Finished loading DATA')
    train = train.drop(train[(train.price < 1.0)].index)
    y = np.log1p(train["price"])
    submission: pd.DataFrame = test[['test_id']]
# ——————————————————————————————————————————————
    handle_missing_inplace(train)
    cutting(train)
    to_categorical(train)

    handle_missing_inplace(test)
    cutting(test)
    to_categorical(test)
# ——————————————————————————————————————————————
    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                                  "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                                  "idf": None,
                                                                  }), procs=8)
    wb.dictionary_freeze= True
    X_name = wb.fit_transform(train['name'])
    X_test_name = wb.transform(test['name'])
    del(wb)
# ——————————————————————————————————————————————
    wb = CountVectorizer()
    X_train_category = wb.fit_transform(train['category_name'])
    X_test_category = wb.transform(test['category_name'])
# ——————————————————————————————————————————————
    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                                  "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                                  "idf": None}), procs=8)
    wb.dictionary_freeze= True
    X_description = wb.fit_transform(train['item_description'])
    X_test_description = wb.transform(test['item_description'])
    del(wb)
# ——————————————————————————————————————————————
    lb = LabelBinarizer(sparse_output=True)
    X_train_brand = lb.fit_transform(train['brand_name'])
    X_test_brand = lb.transform(test['brand_name'])
# ——————————————————————————————————————————————
    X_train_dummies = csr_matrix(pd.get_dummies(train[['item_condition_id', 'shipping']], sparse=True).values)
    X_test_dummies = csr_matrix(pd.get_dummies(test[['item_condition_id', 'shipping']], sparse=True).values)

    del train, test
    gc.collect()
# ——————————————————————————————————————————————
    sparse_train = hstack((X_train_dummies, X_description, X_train_brand, X_train_category, X_name)).tocsr()
    sparse_test = hstack((X_test_dummies, X_test_description, X_test_brand, X_test_category, X_test_name)).tocsr()
# Remove features with document frequency <= 1
    mask = np.array(np.clip(sparse_train.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    X = sparse_train[:, mask]
    X_test = sparse_test[:, mask]
    if develop:
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.01, random_state=42)
    else:
        train_X, train_y = X, y
    print('Preprocessing and Feature Engineering - DONE.')
# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=X.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                    D_fm=200, e_noise=0.0001, iters=20, inv_link="identity", threads=4)
    model.fit(train_X, train_y)

    if develop:
        dev_preds_FM = model.predict(valid_X)
        print("FM dev RMSLE:", rmsle(valid_y, dev_preds_FM))

    predsFM = model.predict(X_test)
# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    params = {'learning_rate': 0.65,
              'metric': 'RMSE',
              'nthread': 4,
              'data_random_seed': 2,
              'max_bin':31,
              }

# Remove features with document frequency <= 50
    mask = np.array(np.clip(X.getnnz(axis=0) - 50, 0, 1), dtype=bool)
    X_ = X[:, mask]
    X_test = X_test[:, mask]
    if develop:
        train_X, valid_X, train_y, valid_y = train_test_split(X_, y, test_size=0.01, random_state=42)
    else:
        train_X, train_y = X_, y

    d_train = lgb.Dataset(train_X, label=train_y)
    if develop:
        d_valid = lgb.Dataset(valid_X, label=valid_y)
        watchlist = [d_train, d_valid]
    else:
        watchlist = [d_train]

    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=3200, #change from 3400
                      valid_sets=watchlist,
                      early_stopping_rounds=1000,
                      verbose_eval=1000
                      )

    if develop:
        dev_preds_LGB = model.predict(valid_X)
        print("LGB dev RMSLE:", rmsle(valid_y, dev_preds_LGB))

    predsLGB = model.predict(X_test)
# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    if develop:
        best = 0
        lowest = 0.999
        for i in range(1000):
            r = i*0.001
            dev_preds_mix = best_ratio_finder(dev_preds_FM, dev_preds_LGB, r)
            score = rmsle(valid_y, dev_preds_mix)
            if score < lowest:
                best = r
                lowest = score
        dev_preds_mix = best_ratio_finder(dev_preds_FM, dev_preds_LGB, best)
        print('BEST weight for [FM, LGB] on dev is: [', best, ', ', 1.0 - best, ']. ')
        print("BEST RMSL-E for [FM, LGB] on dev is: ", rmsle(valid_y, dev_preds_mix))
        preds_final = best_ratio_finder(predsFM, predsLGB, best)
    else:
        preds_final = best_ratio_finder(predsFM, predsLGB, 0.6436)

    submission['price'] = np.clip(np.expm1(preds_final), 0, None)
    submission.to_csv("fm_lgb_no_merge.csv", index=False)

if __name__ == '__main__':
    main()
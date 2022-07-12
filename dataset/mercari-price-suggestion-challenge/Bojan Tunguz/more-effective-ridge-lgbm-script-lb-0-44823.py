import gc
import time
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb

NUM_BRANDS = 4004
NUM_CATEGORIES = 1001
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 3


def handle_missing_inplace(dataset):
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['category_name'].isin(pop_category), 'category_name'] = 'missing'


def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


def main():
    start_time = time.time()

    train = pd.read_table('../input/train.tsv', engine='c')
    test = pd.read_table('../input/test.tsv', engine='c')
    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)

    nrow_train = train.shape[0]
    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, test])
    submission: pd.DataFrame = test[['test_id']]

    del train
    del test
    gc.collect()

    handle_missing_inplace(merge)
    print('[{}] Finished to handle missing'.format(time.time() - start_time))

    cutting(merge)
    print('[{}] Finished to cut'.format(time.time() - start_time))

    to_categorical(merge)
    print('[{}] Finished to convert categorical'.format(time.time() - start_time))

    cv = CountVectorizer(min_df=NAME_MIN_DF)
    X_name = cv.fit_transform(merge['name'])
    print('[{}] Finished count vectorize `name`'.format(time.time() - start_time))

    cv = CountVectorizer()
    X_category = cv.fit_transform(merge['category_name'])
    print('[{}] Finished count vectorize `category_name`'.format(time.time() - start_time))

    tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 3),
                         stop_words='english')
    X_description = tv.fit_transform(merge['item_description'])
    print('[{}] Finished TFIDF vectorize `item_description`'.format(time.time() - start_time))

    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name'])
    print('[{}] Finished label binarize `brand_name`'.format(time.time() - start_time))

    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
    print('[{}] Finished to get dummies on `item_condition_id` and `shipping`'.format(time.time() - start_time))

    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()
    print('[{}] Finished to create sparse merge'.format(time.time() - start_time))

    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_train:]
    
    #train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.1, random_state = 144) 
    d_train = lgb.Dataset(X, label=y)
    #d_valid = lgb.Dataset(valid_X, label=valid_y, max_bin=8192)
    #watchlist = [d_train, d_valid]
    
    params = {
        'learning_rate': 0.75,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 100,
        'verbosity': -1,
        'metric': 'RMSE',
    }


    model = lgb.train(params, train_set=d_train, num_boost_round=3200, verbose_eval=100) 
    preds = 0.57*model.predict(X_test)


    model = Ridge(solver="sag", fit_intercept=True, random_state=205)
    model.fit(X, y)
    print('[{}] Finished to train ridge'.format(time.time() - start_time))
    preds += 0.43*model.predict(X=X_test)
    print('[{}] Finished to predict ridge'.format(time.time() - start_time))


    submission['price'] = np.expm1(preds)
    submission.to_csv("submission_lgbm_ridge_5.csv", index=False)

if __name__ == '__main__':
    main()
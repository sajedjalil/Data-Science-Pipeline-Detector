from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import scipy
import time
import re
import gc
import sys

import lightgbm as lgb


def transform_category_name(category_name):
    try:
        cat_list = category_name.split('/')
        c1 = cat_list[0]
        c2 = cat_list[1]
        c3 = cat_list[2]
        return c1, c2, c3
    except IndexError:
        if len(cat_list) == 2:
            return cat_list[0], cat_list[1], 'missing'
        elif len(cat_list) == 1:
            return cat_list[0], 'missing', 'missing'
        else:
            return 'missing', 'missing', 'missing'
    except Exception:
        return  'missing', 'missing', 'missing'


def extract_target(df):
    df = df.drop(df[df.price <= 1.0].index)
    y = np.log1p(df.price)
    return df, y


def merge_dataset(df, test):
    nrow_train = df.shape[0]
    df_merge = pd.concat([df, test])
    return df_merge, nrow_train


def handle_missing(df):
    # deal with missing data
    df['item_description'].fillna(value='No description yet', inplace=True)
    df['category_name'].fillna(value='missing', inplace=True)
    df['brand_name'].fillna(value='missing', inplace=True)
    return df


def normalize_high_weight_words(df):
    to_replace = [
        r'(\d+)(\.)(\d+)',      # 1.5 ml -> 1`5 ml
        r'(\d+)(\s+)?[gG][bB]?\s+', # 16 gb, 16GB, 16 g -> 16g
        r'(\d+)(\s+)?[tT][bB]?\s+', # 1 tb, 1TB, 1 T -> 1t
        #r'(\d+(`\d+)?)\s+[mM][lL][\s\W]+', # "5`32 ml. 5 ml. 5`0 ml." -> 5`32ml
        #r'(\d+(`\d+)?)(\s+)?([fF][lL](uit)?)?(\s+)?[oO][zZ][\s\W]+', # 5`32 fluit oz/4 fl oz/5`0 oz/8fl oz -> 5`32oz
        r'\s+[tT][\s+|-][Ss][hH][iI][rR][tT]', # t shirt/t-shirt -> tshirt
        r'(\d+)kt\s+', # 14kt -> 14k; this is for Jewelry products
        r'\s+S925\s+', # S925 -> 925; this is for Jewelry products
    ]
    value = [
        r'\1`\3',
        r'\1g ',
        r'\1t ',
        #r'\1ml ',
        #r'\1oz ',
        ' tshirt',
        r'\1k ',
        r' 925 ',
    ]
    df.replace(to_replace=to_replace, value=value, inplace=True, regex=True)
    return df


def extract_addtional_features(df):
    # dense feature: has_description
    df['has_description'] = 1
    df.loc[df['item_description']=='No description yet', 'has_description'] = 0
    # dense feature: has_price
    df['has_price'] = 0
    df.loc[df['item_description'].str.contains('[rm]', regex=False), 'has_price'] = 1
    df.loc[df['name'].str.contains('[rm]', regex=False), 'has_price'] = 1
    return df


def handle_categories(df):
    df['c1'], df['c2'], df['c3'] = zip(*df['category_name'].apply(transform_category_name))
    df.drop('category_name', axis=1, inplace=True)
    df['c1'] = df['c1'].astype('category')
    df['c2'] = df['c2'].astype('category')
    df['c3'] = df['c3'].astype('category')
    df['item_condition_id'] = df['item_condition_id'].astype('category')
    return df


def extract_all_features(df, nrow_train, start=time.time()):
    cv = CountVectorizer(min_df=10, max_df=0.9, lowercase=True, max_features=60000,
                         analyzer='word', token_pattern=r'[\w`]+', stop_words='english')
    X_name = cv.fit_transform(df.name)
    print('[{:.2f}] Trnasform name data completed. X_name={}'.format(time.time() - start, X_name.shape))

    cv = CountVectorizer(min_df=10)
    X_c1 = cv.fit_transform(df.c1)
    X_c2 = cv.fit_transform(df.c2)
    X_c3 = cv.fit_transform(df.c3)
    print('[{:.2f}] Transform category data completed. X_c1={}, X_c2={}, X_c3={}'.format(time.time() - start, X_c1.shape, X_c2.shape, X_c3.shape))

    tv = TfidfVectorizer(min_df=10, max_df=0.9, lowercase=True, max_features=100000,
                         analyzer='word', token_pattern=r'[\w`]+', use_idf=True,
                         smooth_idf=True, sublinear_tf=True, stop_words='english')
    X_description = tv.fit_transform(df.item_description)
    print('[{:.2f}] Trnasform item description data completed. X_description={}'.format(time.time() - start, X_description.shape))

    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(df.brand_name)
    print('[{:.2f}] Trnasform brand data completed. X_brand={}'.format(time.time() - start, X_brand.shape))

    X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(df[['item_condition_id', 'shipping',
                                                           'has_description', 'has_price']], sparse=True).values)
    print('[{:.2f}] Handle dummies completed. X_dummies={}'.format(time.time() - start, X_dummies.shape))

    X_merge = scipy.sparse.hstack((X_dummies, X_brand, X_description, X_c1, X_c2, X_c3, X_name)).tocsr()
    print('[{:.2f}] Merged features completed. X_merge={}'.format(time.time() - start, X_merge.shape))

    X = X_merge[:nrow_train]
    X_test = X_merge[nrow_train:]
    print('[{:.2f}] Shape of X={}, X_test={}:'.format((time.time() - start), X.shape, X_test.shape))
    return X, X_test


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(y - y0, 2)))


def rmsle_scoring(estimator, X, y):
    y_pred = estimator.predict(X)
    return rmsle(y_pred, y)


def main(has_cv=False, debug=False):
    start = time.time()

    print('[{:.2f}] Begin to read train and test dataset. has_cv={}, debug={}'.format(time.time() - start, has_cv, debug))
    df_raw = pd.read_csv("../input/train.tsv", sep="\t")
    df_test_raw = pd.read_csv("../input/test.tsv", sep="\t")
    print('[{:.2f}] Read dataset completed'.format(time.time() - start))

    if debug:
        # for quick debuging
        df_train = df_raw.sample(frac=0.01, random_state=4, axis=0)
        df_test = df_test_raw.sample(frac=0.01, random_state=4, axis=0)
    else:
        # train by full dataset
        df_train = df_raw
        df_test = df_test_raw

    df_train, y = extract_target(df_train)
    print('[{:.2f}] Extract target and drop invalid price completed'.format(time.time() - start))

    df_merge, nrow_train = merge_dataset(df_train, df_test)
    del df_train
    gc.collect()
    print('[{:.2f}] Merge dataset completed'.format(time.time() - start))

    df_merge = handle_missing(df_merge)
    print('[{:.2f}] Handle missing completed'.format(time.time() - start))

    df_merge = normalize_high_weight_words(df_merge)
    print('[{:.2f}] Normalize high weight words completed'.format(time.time() - start))

    df_merge = extract_addtional_features(df_merge)
    print('[{:.2f}] Extract addtional features completed'.format(time.time() - start))

    df_merge = handle_categories(df_merge)
    print('[{:.2f}] Handle category features completed'.format(time.time() - start))

    X, X_test = extract_all_features(df_merge, nrow_train, start)
    print('[{:.2f}] Extract all features completed'.format(time.time() - start))

    if has_cv:
        train_X, cv_X, train_y, cv_y = train_test_split(X, y, test_size=0.1, random_state=144)
        print('[{:.2f}] Split train and cross-validation dataset completed'.format(time.time() - start))

    print('[{:.2f}] LGBMRegressor train start ...'.format(time.time() - start))
    lgbm_model = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',
                                   random_state=11, n_jobs=4, subsample_for_bin=81920,
                                   learning_rate=0.2, num_leaves=140, max_depth=40, n_estimators=2200)

    if has_cv:
        lgbm_model.fit(train_X, train_y, early_stopping_rounds=1000, verbose=False,
                       eval_set=[(cv_X, cv_y)], eval_metric="rmse",
                       callbacks=[lgb.print_evaluation(period=100)])
        train_score = rmsle_scoring(lgbm_model, train_X, train_y)
    else:
        lgbm_model.fit(X, y, verbose=False, eval_metric="rmse",
                       callbacks=[lgb.print_evaluation(period=100)])
        train_score = rmsle_scoring(lgbm_model, X, y)
    print('[{:.2f}] LGBMRegressor train completed. Train score: {:.6f}'.format(time.time() - start, train_score))

    predsL = lgbm_model.predict(X_test)
    print('[{:.2f}] LGBMRegressor predict completed.'.format(time.time() - start))

    df_test['price'] = np.expm1(predsL)
    df_test[['test_id', 'price']].to_csv("submission_lgbm_1.csv", index=False)
    print('[{:.2f}] Submission completed.'.format(time.time() - start))

if __name__ == '__main__':
    main(has_cv=False, debug=False)
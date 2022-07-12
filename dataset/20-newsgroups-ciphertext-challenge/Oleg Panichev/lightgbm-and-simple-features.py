import datetime
import gc
import numpy as np
import os
import pandas as pd
import random

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import skew, kurtosis

import Levenshtein
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import lightgbm as lgb

from tqdm import tqdm


id_col = 'Id'
target_col = 'target'

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

def extract_features(df):
    df['nunique'] = df['ciphertext'].apply(lambda x: len(np.unique(x)))
    df['len'] = df['ciphertext'].apply(lambda x: len(x))

    def count_chars(x):
        n_l = 0 # count letters
        n_n = 0 # count numbers
        n_s = 0 # count symbols
        n_ul = 0 # count upper letters
        n_ll = 0 # count lower letters
        for i in range(0, len(x)):
            if x[i].isalpha():
                n_l += 1
                if x[i].isupper():
                    n_ul += 1
                elif x[i].islower():
                    n_ll += 1
            elif x[i].isdigit():
                n_n += 1
            else:
                n_s += 1

        return pd.Series([n_l, n_n, n_s, n_ul, n_ll])

    cols = ['n_l', 'n_n', 'n_s', 'n_ul', 'n_ll']
    for c in cols:
        df[c] = 0
    tqdm.pandas(desc='count_chars')
    df[cols] = df['ciphertext'].progress_apply(lambda x: count_chars(x))
    for c in cols:
        df[c] /= df['len']

    tqdm.pandas(desc='distances')
    df['Levenshtein_distance'] = df['ciphertext'].progress_apply(lambda x: Levenshtein.distance(x, x[::-1]))
    df['Levenshtein_ratio'] = df['ciphertext'].progress_apply(lambda x: Levenshtein.ratio(x, x[::-1]))
    df['Levenshtein_jaro'] = df['ciphertext'].progress_apply(lambda x: Levenshtein.jaro(x, x[::-1]))
    df['Levenshtein_hamming'] = df['ciphertext'].progress_apply(lambda x: Levenshtein.hamming(x, x[::-1]))

    for m in range(1, 5):
        df['Levenshtein_distance_m{}'.format(m)] = df['ciphertext'].progress_apply(lambda x: Levenshtein.distance(x[:-m], x[m:]))
        df['Levenshtein_ratio_m{}'.format(m)] = df['ciphertext'].progress_apply(lambda x: Levenshtein.ratio(x[:-m], x[m:]))
        df['Levenshtein_jaro_m{}'.format(m)] = df['ciphertext'].progress_apply(lambda x: Levenshtein.jaro(x[:-m], x[m:]))
        df['Levenshtein_hamming_m{}'.format(m)] = df['ciphertext'].progress_apply(lambda x: Levenshtein.hamming(x[:-m], x[m:]))
    
    df['Levenshtein_distance_h'] = df['ciphertext'].progress_apply(lambda x: Levenshtein.distance(x[:len(x)//2], x[len(x)//2:]))
    df['Levenshtein_ratio_h'] = df['ciphertext'].progress_apply(lambda x: Levenshtein.ratio(x[:len(x)//2], x[len(x)//2:]))
    df['Levenshtein_jaro_h'] = df['ciphertext'].progress_apply(lambda x: Levenshtein.jaro(x[:len(x)//2], x[len(x)//2:]))
    
    # All symbols stats
    def strstat(x):
        r = np.array([ord(c) for c in x])
        return pd.Series([
            np.sum(r), 
            np.mean(r), 
            np.std(r), 
            np.min(r), 
            np.max(r),
            skew(r), 
            kurtosis(r),
            ])
    cols = ['str_sum', 'str_mean', 'str_std', 'str_min', 'str_max', 'str_skew', 'str_kurtosis']
    for c in cols:
        df[c] = 0
    tqdm.pandas(desc='strstat')
    df[cols] = df['ciphertext'].progress_apply(lambda x: strstat(x))
    
    # Digit stats
    def str_digit_stat(x):
        r = np.array([ord(c) for c in x if c.isdigit()])
        if len(r) == 0:
            r = np.array([0])
        return pd.Series([
            np.sum(r), 
            np.mean(r), 
            np.std(r), 
            np.min(r), 
            np.max(r),
            skew(r), 
            kurtosis(r),
            ])
    cols = ['str_digit_sum', 'str_digit_mean', 'str_digit_std', 'str_digit_min', 
        'str_digit_max', 'str_digit_skew', 'str_digit_kurtosis']
    for c in cols:
        df[c] = 0
    tqdm.pandas(desc='str_digit_stat')
    df[cols] = df['ciphertext'].progress_apply(lambda x: str_digit_stat(x))
    

print('Extracting features for train:')
extract_features(train)
print('Extracting features for test:')
extract_features(test)

# TFIDF
for k in range(0, 3):
    tfidf = TfidfVectorizer(
        max_features=1000,
        lowercase=False,
        token_pattern='\\S+',
    )

    def char_pairs(x, k=1):
        buf = []
        for i in range(k, len(x)):
            buf.append(x[i-k:i+1])
        return ' '.join(buf)

    train['text_temp'] = train.ciphertext.apply(lambda x: char_pairs(x, k))
    test['text_temp'] = test.ciphertext.apply(lambda x: char_pairs(x, k))
    train_tfids = tfidf.fit_transform(train['text_temp'].values).todense()
    test_tfids = tfidf.transform(test['text_temp'].values).todense()

    print('k = {}: train_tfids.shape = {}'.format(k, train_tfids.shape))
    
    for i in range(train_tfids.shape[1]):
        train['text_{}_tfidf{}'.format(k, i)] = train_tfids[:, i]
        test['text_{}_tfidf{}'.format(k, i)] = test_tfids[:, i]

    del train_tfids, test_tfids, tfidf
    gc.collect()

# Build the model
cnt = 0
p_buf = []
p_valid_buf = []
n_splits = 5
kf = KFold(
    n_splits=n_splits, 
    random_state=0)
err_buf = []   
undersampling = 0

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'max_depth': 5,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': -1,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0,
    'min_gain_to_split': 0,
    'num_class': train[target_col].nunique(),
}

cols_to_drop = [
    id_col, 
    'ciphertext',
    target_col,
    'text_temp',
]

X = train.drop(cols_to_drop, axis=1, errors='ignore')
feature_names = list(X.columns)

X = X.values
y = train[target_col].values

X_test = test.drop(cols_to_drop, axis=1, errors='ignore')
id_test = test[id_col].values

print(X.shape, y.shape)
print(X_test.shape)

n_features = X.shape[1]

for train_index, valid_index in kf.split(X, y):
    print('Fold {}/{}'.format(cnt + 1, n_splits))
    params = lgb_params.copy() 

    lgb_train = lgb.Dataset(
        X[train_index], 
        y[train_index], 
        feature_name=feature_names,
        )
    lgb_train.raw_data = None

    lgb_valid = lgb.Dataset(
        X[valid_index], 
        y[valid_index],
        )
    lgb_valid.raw_data = None

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=10000,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    if cnt == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        for i in range(20):
            if i < len(tuples):
                print(tuples[i])
            else:
                break

        del importance, model_fnames, tuples

    p = model.predict(X[valid_index], num_iteration=model.best_iteration)
    err = f1_score(y[valid_index], np.argmax(p, axis=1), average='macro')

    print('{} F1: {}'.format(cnt + 1, err))

    p = model.predict(X_test, num_iteration=model.best_iteration)
    if len(p_buf) == 0:
        p_buf = np.array(p, dtype=np.float16)
    else:
        p_buf += np.array(p, dtype=np.float16)
    err_buf.append(err)

    cnt += 1

    del model, lgb_train, lgb_valid, p
    gc.collect

    # Train on one fold
    if cnt > 0:
        break


err_mean = np.mean(err_buf)
err_std = np.std(err_buf)
print('F1 = {:.6f} +/- {:.6f}'.format(err_mean, err_std))

preds = p_buf/cnt

# Prepare submission
subm = pd.DataFrame()
subm[id_col] = id_test
subm['Predicted'] = np.argmax(preds, axis=1)
subm.to_csv('submission.csv', index=False)


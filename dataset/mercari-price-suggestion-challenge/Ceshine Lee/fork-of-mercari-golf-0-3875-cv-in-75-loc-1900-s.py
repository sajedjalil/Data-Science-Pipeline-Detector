import gc
import os; os.environ['OMP_NUM_THREADS'] = '1'
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
from typing import List, Dict

import keras as ks
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from scipy.sparse import vstack

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')
    df['text'] = (df['item_description'].fillna('') + ' ' + df['name'] + ' ' + df['category_name'].fillna(''))
    return df[['name', 'text', 'shipping', 'item_condition_id']]

def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

def fit_predict(xs, y_train) -> np.ndarray:
    X_train, X_test = xs
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(graph=tf.Graph(), config=config) as sess, timer('fit_predict'):
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(192, activation='relu')(model_in)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1)(out)
        model = ks.Model(model_in, out)
        model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
        for i in range(3):
            with timer(f'epoch {i + 1}'):
                model.fit(x=X_train, y=y_train, batch_size=2**(11 + i), epochs=1, verbose=0)
        return model.predict(X_test, batch_size=2**(11 + i))[:, 0]

def main():
    vectorizer = make_union(
        on_field('name', Tfidf(max_features=100000, token_pattern='\w+')),
        on_field('text', Tfidf(max_features=100000, token_pattern='\w+', ngram_range=(1, 2))),
        on_field(['shipping', 'item_condition_id'],
                 FunctionTransformer(to_records, validate=False), DictVectorizer()),
        n_jobs=4)
    y_scaler = StandardScaler()
    with timer('process train'):
        train = pd.read_table('../input/train.tsv')
        train = train[train['price'] > 0].reset_index(drop=True)
        cv = KFold(n_splits=20, shuffle=True, random_state=9705)
        train_ids, valid_ids = next(cv.split(train))
        train, valid = train.iloc[train_ids], train.iloc[valid_ids]
        y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
        X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
        print(f'X_train: {X_train.shape} of {X_train.dtype}')
        del train
    with timer('process valid'):
        X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)
    gc.collect()
    print(X_train.shape)
    print(X_valid.shape)
    with timer('process test'):
        test = pd.read_table('../input/test_stg2.tsv')
        X_test_tmp = []
        for i in range(0, len(test), 500000):
            print("Test {} to {}".format(i, i+500000))
            X_test_tmp.append(
                vectorizer.transform(
                    preprocess(test.iloc[i:(i+500000)].copy())
                ).astype(np.float32)
            )
        X_test = vstack(X_test_tmp)
        del test, X_test_tmp
        gc.collect()
    print(X_test.shape)
    valid_length = X_valid.shape[0]
    X_valid = vstack([X_valid, X_test])
    with ThreadPool(processes=4) as pool:
        Xb_train, Xb_valid = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_valid]]
        xs = [[Xb_train, Xb_valid], [X_train, X_valid]] * 2
        y_pred = np.mean(pool.map(partial(fit_predict, y_train=y_train), xs), axis=0)
    y_pred = np.expm1(y_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0])
    test_pred = y_pred[valid_length:]
    y_pred = y_pred[:valid_length]
    print('Valid RMSLE: {:.4f}'.format(np.sqrt(mean_squared_log_error(valid['price'], y_pred))))
    submission = pd.read_csv(
        f'../input/test_stg2.tsv', sep='\t', usecols=["test_id"])
    submission["price"] = test_pred
    submission.to_csv("sub.csv",
                      index=False, float_format="%.8f")
                      
if __name__ == '__main__':
    main()

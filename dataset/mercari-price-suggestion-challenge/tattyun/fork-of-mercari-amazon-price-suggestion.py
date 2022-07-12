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

#実行時間計測
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

    #データフレームの欠損値埋め
    #商品名とブランド名を結合。TF-IDFしやすくするためtextという要素作成。
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')
    df['text'] = (df['name'] + ' ' + df['category_name'].fillna(''))
    #df['text'] = (df['item_description'].fillna('') + ' ' + df['name'] + ' ' + df['category_name'].fillna(''))
    return df[['name', 'text', 'shipping', 'item_condition']]
    #return df[['name', 'text', 'shipping', 'item_condition_id']]

#make_pipelineでitemgetterとTfidfのインスタンスをパイプライン化している。
#FunctionTransformerでitemgetterをtransformerに変換して自前の変換器を作成している。
#これにより、itemgetter（文字列の抽出）の重要な文字列の特定を一連の流れでこの後行えるようになっている。
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
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)#MLPの作成
        out = ks.layers.Dense(192, activation='relu')(model_in)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1)(out)
        model = ks.Model(model_in, out)
        model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
        for i in range(3):#3エポック
            with timer(f'epoch {i + 1}'):
                model.fit(x=X_train, y=y_train, batch_size=2**(11 + i), epochs=1, verbose=0)#バッチサイズは指数関数的に増加させる
        return model.predict(X_test)[:, 0]#予想を返す

#on_field()でname,textの要素抽出、TFIDFで数値化
def main():
    vectorizer = make_union(
        on_field('name', Tfidf(max_features=100000, token_pattern='\w+')),
        on_field('text', Tfidf(max_features=100000, token_pattern='\w+', ngram_range=(1, 2))),
        #on_field(['shipping', 'item_condition'],
        #         FunctionTransformer(to_records, validate=False), DictVectorizer()),n_jobs=4)
        on_field(['shipping', 'item_condition'],
                 FunctionTransformer(to_records, validate=False), DictVectorizer()))
        #on_field(['shipping', 'item_condition_id'],
         #        FunctionTransformer(to_records, validate=False), DictVectorizer()),
       # n_jobs=4)
    y_scaler = StandardScaler()
#教師データの作成
    with timer('process train'):
        #train = pd.read_table('../input/mercari/train.tsv')
    #ロード
        #train = pd.read_table('../input/mercariori/mercariData.tsv')
        #train = pd.read_table('../input/amazonorigin/Amazon.tsv')
        #train = pd.read_table('../input/amazonor/Amazon32.tsv')
        #train = pd.read_table('../input/amazono46/amazon46.tsv')
        #train = pd.read_table('../input/amazo19/Amazon66.tsv')
        #train = pd.read_table('../input/amazo19/AmazonBooks.tsv')
        #train = pd.read_table('../input/amazo19/AmazonGame.tsv')
        #train = pd.read_table('../input/amazo19/AmazonGameBooks.tsv')
        #train = pd.read_table('../input/amazo21/AmazonGameBooksPhone.tsv')
        #train = pd.read_table('../input/amazo21/Amazon86.tsv')
        #train = pd.read_table('../input/amazo21/AmazonPhone.tsv')
        #train = pd.read_table('../input/amazo22/AmazonGamePhone.tsv')
        #train = pd.read_table('../input/amazo22/AmazonBooksPhone.tsv')
        #train = pd.read_table('../input/amazo1104/AmazonGameBooksPhone2.tsv')
        #train = pd.read_table('../input/amazo1105/AmazonGame2.tsv')
        #train = pd.read_table('../input/amazo1112/AmazonGameBooksPhone3.tsv')
        #train = pd.read_table('../input/amazo1112/AmazonGame33.tsv')
        train = pd.read_table('../input/amazo1227/AmazonGameBooksPhone4.tsv')
        
    #0ドルのpriceが存在するためはじいている
        train = train[train['price'] > 0].reset_index(drop=True)
#simに類似度推定で求めた上位5までのindexを入れる->求めたいもの(index=0)のみ
        #sim1 = ([19,  1, 17,  2,  5])
        #sim1 = ([20, 17, 71, 4, 2]) # X = 0
        #sim1 = ([71,  4, 70, 20, 72]) # X=2
        sim1 = ([2, 70, 20, 71, 72]) # X = 4
        sim = ([0])
    #データを学習用と検証用で分割するための準備
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
    #データを学習用と検証用で分割
    #.split()でイテラブルなオブジェクトが帰ってくる。学習用の「インデックスと検証用のインデックスが取り出せる。
    #next()でイテレータ内から要素を取得
        train_ids, valid_ids = next(cv.split(train))
    #取得したインデックスで学習と検証用に分割
        train, valid, tes ,val_pri= train.iloc[train_ids], train.iloc[valid_ids], train.iloc[sim], train.iloc[sim1]
#tes追加
        #tes = train.iloc[sim]
    #価格は1行n列をn行1列に変換。log(a+1)で変換。正規化
        y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
    #パイプラインで処理
        X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
        print(f'X_train: {X_train.shape} of {X_train.dtype}')
        del train
        #検証データも同様に前処理
    with timer('process valid'):
        X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)
        X_tes = vectorizer.transform(preprocess(tes)).astype(np.float32)
    with ThreadPool(processes=4) as pool: #4つのスレッドにする
        Xb_train, Xb_valid = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_valid]]
        xs = [[Xb_train, Xb_valid], [X_train, X_valid]] * 2
        #partial()でfit_predict内のy_trainを固定、xsの値だけ変えている
        y_pred = np.mean(pool.map(partial(fit_predict, y_train=y_train), xs), axis=0)#4コアで学習したものの平均をとっている
    y_pred = np.expm1(y_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0])#logで変換したものを価格に直す
    print('Valid RMSLE: {:.4f}'.format(np.sqrt(mean_squared_log_error(valid['price'], y_pred))))
    #print('\n',y_pred)
    #print('\n',valid['price'])
    #print(X_valid)
    with ThreadPool(processes=4) as pool: #4つのスレッドにする
        Xb_train, Xb_tes= [x.astype(np.bool).astype(np.float32) for x in [X_train, X_tes]]
        xs = [[Xb_train, Xb_tes], [X_train, X_tes]] * 2
        #partial()でfit_predict内のy_trainを固定、xsの値だけ変えている
        y_tes = np.mean(pool.map(partial(fit_predict, y_train=y_train), xs), axis=0)#4コアで学習したものの平均をとっている
    y_tes = np.expm1(y_scaler.inverse_transform(y_tes.reshape(-1, 1))[:, 0])#logで変換したものを価格に直す
    #y_tes = fit_predict(sim, y_train)
    print('Valid RMSLE: {:.4f}'.format(np.sqrt(mean_squared_log_error(valid['price'], y_pred))))
    print('\n',y_tes) #simのindexの推定価格
    #print('\n',valid['price'])
    #print(X_tes)
    print(val_pri['price']) #類似アイテムの実価格
    
    for pri in val_pri['price']: #推定価格と類似度上位アイテムの実価格の差が実価格の20%以内なら推薦。超えてたら次の順位チェック
        if abs(pri - y_tes) < (pri*0.20): #0.47
            print(pri)
            print(val_pri[val_pri['price']== pri ].index)
        #print(abs(price - y_tes))
            break #5個全て条件外だったら-1を返してレコメンドなし
            
if __name__ == '__main__':
    main()
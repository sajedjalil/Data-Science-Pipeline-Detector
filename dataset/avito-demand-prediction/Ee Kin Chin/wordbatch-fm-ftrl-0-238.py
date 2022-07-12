# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import sys

sys.path.insert(0, '../input/wordbatch-133/wordbatch/')
sys.path.insert(0, '../input/randomstate/randomstate/')
import wordbatch
from sklearn.metrics import mean_squared_error
from math import sqrt
from wordbatch.extractors import WordHash
from wordbatch.models import FM_FTRL
from wordbatch.data_utils import *
import threading
import pandas as pd
from sklearn.metrics import roc_auc_score
import time
import numpy as np
import gc
from contextlib import contextmanager
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import regex as re
from scipy.sparse import hstack
from nltk.corpus import stopwords
stopWords = stopwords.words('russian')

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

import os, psutil

start_time = time.time()


def df2csr(df, traindex, testdex, len_train):
    df.reset_index(drop=True, inplace=True)
    with timer("Adding Time Variables"):
        df["Year"] = df["activation_date"].dt.year
        df["Date of Year"] = df['activation_date'].dt.dayofyear # Day of Year
        df["Weekday"] = df['activation_date'].dt.weekday
        df["Weekd of Year"] = df['activation_date'].dt.week
        df["Day of Month"] = df['activation_date'].dt.day
        df["Quarter"] = df['activation_date'].dt.quarter
        df.drop(["activation_date","image"],axis=1,inplace=True)

    with timer("Encode Variables"):
        categorical = ["user_id","region","city","parent_category_name","category_name","item_seq_number","user_type"]
        messy_categorical = ["param_1","param_2","param_3"] # Need to find better technique for these
        # Encoder:
        lbl = preprocessing.LabelEncoder()
        for col in categorical + messy_categorical:
            df[col] = lbl.fit_transform(df[col].astype(str))
        del lbl
        gc.collect()

    with timer("Log features"):
        for fea in ['price','image_top_1']:
            df[fea]= np.log2(1 + df[fea].values).astype(int)

    with timer("splitting to train and test"):
        trn = df[:len_train].copy()
        print("Training Set shape",trn.shape)
        tst = df[len_train:].copy()
        print("Submission Set Shape: {} Rows, {} Columns".format(*tst.shape))
        del df
        gc.collect()

    with timer("Generating str_array"):
        trn_str_array = ("Y" + trn['Year'].astype(str) \
            + " DY" + trn['Date of Year'].astype(str) \
            + " WD" + trn['Weekday'].astype(str) \
            + " WDY" + trn['Weekd of Year'].astype(str) \
            + " DM" + trn['Day of Month'].astype(str) \
            + " Q" + trn['Quarter'].astype(str) \
            + " UI" + trn['user_id'].astype(str) \
            + " R" + trn['region'].astype(str) \
            + " C" + trn['city'].astype(str) \
            + " PCN" + trn['parent_category_name'].astype(str) \
            + " CN" + trn['category_name'].astype(str) \
            + " ISN" + trn['item_seq_number'].astype(str) \
            + " UT" + trn['user_type'].astype(str) \
            + " PO" + trn['param_1'].astype(str) \
            + " PT" + trn['param_2'].astype(str) \
            + " PTT" + trn['param_3'].astype(str) \
            + " T" + trn['title'].astype(str) \

            + " UXC" + trn['user_id'].astype(str)+"_"+trn['city'].astype(str) \
            + " RXC" + trn['region'].astype(str)+"_"+trn['city'].astype(str) \
            + " CXI" + trn['category_name'].astype(str)+"_"+trn['item_seq_number'].astype(str) \
            + " UXU" + trn['user_type'].astype(str)+"_"+trn['user_id'].astype(str) \
            + " PXI" + trn['parent_category_name'].astype(str)+"_"+trn['item_seq_number'].astype(str) \

            + " P" + trn['price'].astype(str) \
            + " IT" + trn['image_top_1'].astype(str)).values
        tst_str_array = ("Y" + tst['Year'].astype(str) \
            + " DY" + tst['Date of Year'].astype(str) \
            + " WD" + tst['Weekday'].astype(str) \
            + " WDY" + tst['Weekd of Year'].astype(str) \
            + " DM" + tst['Day of Month'].astype(str) \
            + " Q" + tst['Quarter'].astype(str) \
            + " UI" + tst['user_id'].astype(str) \
            + " R" + tst['region'].astype(str) \
            + " C" + tst['city'].astype(str) \
            + " PCN" + tst['parent_category_name'].astype(str) \
            + " CN" + tst['category_name'].astype(str) \
            + " ISN" + tst['item_seq_number'].astype(str) \
            + " UT" + tst['user_type'].astype(str) \
            + " PO" + tst['param_1'].astype(str) \
            + " PT" + tst['param_2'].astype(str) \
            + " PTT" + tst['param_3'].astype(str) \
            + " T" + tst['title'].astype(str) \

            + " UXC" + tst['user_id'].astype(str)+"_"+tst['city'].astype(str) \
            + " RXC" + tst['region'].astype(str)+"_"+tst['city'].astype(str) \
            + " CXI" + tst['category_name'].astype(str)+"_"+tst['item_seq_number'].astype(str) \
            + " UXU" + tst['user_type'].astype(str)+"_"+tst['user_id'].astype(str) \
            + " PXI" + tst['parent_category_name'].astype(str)+"_"+tst['item_seq_number'].astype(str) \

            + " P" + tst['price'].astype(str) \
            + " IT" + tst['image_top_1'].astype(str)).values
    del trn
    gc.collect()
    del tst
    gc.collect()
    return trn_str_array , tst_str_array

def char_analyzer(text):
    """
    This is used to split strings in small lots
    anttip saw this in an article 
    so <talk> and <talking> would have <Tal> <alk> in common
    should be similar to russian I guess
    """
    tokens = text.split()
    return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("\nData Load Stage")
training = pd.read_csv('../input/avito-demand-prediction/train.csv', index_col = "item_id", parse_dates = ["activation_date"])
len_train = len(training)
traindex = training.index
testing = pd.read_csv('../input/avito-demand-prediction/test.csv', index_col = "item_id", parse_dates = ["activation_date"])
testdex = testing.index
labels = training['deal_probability'].values
training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))
# Combine Train and Test
df_all = pd.concat([training,testing],axis=0)
trnshape = training.shape[0]
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df_all.shape))

df_all['description'] = df_all['description'].fillna("")

with timer("Tfidf on word"):
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        tokenizer=lambda x: re.findall(r'[^\p{P}\W]+', x),
        analyzer='word',
        token_pattern=None,
        stop_words=stopWords,
        ngram_range=(1, 2), 
        max_features=5000)
    X = word_vectorizer.fit_transform(df_all['description'])
    train_word_features = X[:trnshape]
    test_word_features = X[trnshape:]
    del (X)

with timer("Tfidf on char n_gram"):
    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        tokenizer=char_analyzer,
        analyzer='word',
        ngram_range=(1, 2),
        max_features=5000)
    X = char_vectorizer.fit_transform(df_all['description'])
    train_char_features = X[:trnshape]
    test_char_features = X[trnshape:]
    del (X)	
with timer("Transforming other features"):  
    trn_str_arrays, tst_str_arrays = df2csr(df_all,  traindex, testdex, len_train)
    del df_all
    gc.collect

    batchsize = 10000000
    D = 2 ** 4

    wb = wordbatch.WordBatch(None, extractor=(WordHash, {"ngram_range": (1, 1), "analyzer": "word",
    													 "lowercase": False, "n_features": D,
    													 "norm": None, "binary": True})
    						 , minibatch_size=batchsize // 80, procs=8, freeze=True, timeout=1800, verbose=0)
    						 
    trn_str_array = wb.transform(trn_str_arrays)
    del trn_str_arrays
    gc.collect()
    tst_str_array = wb.transform(tst_str_arrays)
    del tst_str_arrays
    gc.collect()
    del wb
    gc.collect()

print("DONE")
with timer("Stacking Features"):
    train_features = hstack(
        [
            train_char_features,
            train_word_features
        ]
    )
    del train_word_features, train_char_features
    
    
    train_features = hstack(
        [
            train_features,
            trn_str_array
        ]
    ).tocsr()
    del trn_str_array
    gc.collect()
    print("Fin Stack Train")
    test_features = hstack(
        [
            test_char_features,
            test_word_features,
            tst_str_array
        ]
    ).tocsr()
    del test_word_features, tst_str_array, test_char_features
    gc.collect()
    print("Fin Stack test")


with timer("Scoring FM FTRL"):
    clf = FM_FTRL(
        alpha=0.02, beta=0.01, L1=0.00001, L2=30.0,
        D=train_features.shape[1], alpha_fm=0.1,
        L2_fm=0.5, init_fm=0.01, weight_fm= 50.0,
        D_fm=200, e_noise=0.0, iters=3,
        inv_link="identity", e_clip=1.0, threads=4, use_avx= 1, verbose=1
            )
    clf.fit(train_features, labels)
    train_pred = sigmoid(clf.predict(train_features))
    pred = sigmoid(clf.predict(test_features))
    score = sqrt(mean_squared_error(labels, train_pred))
    print("FINAL RMSE {}".format(score))
    sub = pd.read_csv('../input/avito-demand-prediction/sample_submission.csv')
    sub['deal_probability'] = y_pred
    sub['deal_probability'].clip(0.0, 1.0, inplace=True)
print("Output Prediction CSV")
sub.to_csv('wordbatch_fmtrl_submission.csv', index=False)
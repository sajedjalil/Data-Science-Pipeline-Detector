import sys
sys.path.append('../input/libftrl-python/')

import ftrl

from time import time

import pandas as pd
import numpy as np
import scipy.sparse as sp

import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

from fastcache import clru_cache as lru_cache

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

@lru_cache(1024)
def stem(s):
    return stemmer.stem(s)
    

stop = {'the', 'was', 'were', 'did', 'had', 'have', 'been', 'will', 'and', 
        'that', 'who', 'are', 'for', 'has'}

tags = re.compile(r'<.+?>')
whitespace = re.compile(r'\s+')
non_letter = re.compile(r'\W+')

def clean_text(text):
    text = text.lower()
    text = non_letter.sub(' ', text)

    tokens = []

    for t in text.split():
        if len(t) <= 2 and not t.isdigit():
            continue
        if t in stop:
            continue
        t = stem(t)
        tokens.append(t)

    text = ' '.join(tokens)

    text = whitespace.sub(' ', text)
    text = text.strip()
    return text

def paths(tokens):
    all_paths = ['/'.join(tokens[0:(i+1)]) for i in range(len(tokens))]
    return ' '.join(all_paths)

@lru_cache(1024)
def cat_process(cat):
    cat = cat.lower()
    cat = whitespace.sub('', cat)
    split = cat.split('/')
    return paths(split)


print('reading train data...')

df_train = pd.read_csv('../input/mercari-price-suggestion-challenge/train.tsv', sep='\t')


price = df_train.pop('price')
y = np.log1p(price.values)

df_train.name.fillna('unkname', inplace=True)
df_train.category_name.fillna('unk_cat', inplace=True)
df_train.brand_name.fillna('unk_brand', inplace=True)
df_train.item_description.fillna('nodesc', inplace=True)


print('processing name & description...')

df_train.category_name = df_train.category_name.apply(cat_process)
df_train.name = df_train.name.apply(clean_text)
df_train.item_description = df_train.item_description.apply(clean_text)

df_train.brand_name = df_train.brand_name.str.lower()
df_train.brand_name = df_train.brand_name.str.replace(' ', '_')


print('OHE...')

cv_name = CountVectorizer(token_pattern='\S+', min_df=10, dtype=np.uint8, ngram_range=(1, 3))
X_name_train = cv_name.fit_transform(df_train.name).astype('float32')

cv_cat = CountVectorizer(token_pattern='\S+', min_df=10, dtype=np.uint8,)
X_cat_train = cv_cat.fit_transform(df_train.category_name)

ohe = OneHotEncoder(dtype=np.uint8)
X_ohe_train = ohe.fit_transform(df_train[['item_condition_id', 'shipping']].values)

cv_brand = CountVectorizer(token_pattern='\S+', min_df=10, dtype=np.uint8)
X_brand_train = cv_brand.fit_transform(df_train.brand_name)

cv_desc = CountVectorizer(token_pattern='\S+', min_df=10, dtype=np.uint8, ngram_range=(1, 3))
X_desc_train = cv_desc.fit_transform(df_train.item_description).astype('float32')

X = sp.hstack([X_name_train, X_cat_train, X_brand_train, X_desc_train, X_ohe_train], format='csr')


print('training FTRL...')

t0 = time()
model = ftrl.FtrlProximal(alpha=0.01, beta=1, l1=75, l2=0, model_type='regression')
model.fit(X, y, num_passes=50)
t1 = time()
took = (t1 - t0) / 60

print('training took %.3f minutes' % took)


print('reading the test data...')

df_test = pd.read_csv('../input/mercari-price-suggestion-challenge/test.tsv', sep='\t')

df_test.name.fillna('unkname', inplace=True)
df_test.category_name.fillna('unk_cat', inplace=True)
df_test.brand_name.fillna('unk_brand', inplace=True)
df_test.item_description.fillna('nodesc', inplace=True)

df_test.category_name = df_test.category_name.apply(cat_process)
df_test.name = df_test.name.apply(clean_text)
df_test.item_description = df_test.item_description.apply(clean_text)
df_test.brand_name = df_test.brand_name.str.lower()
df_test.brand_name = df_test.brand_name.str.replace(' ', '_')


X_name_test = cv_name.transform(df_test.name).astype('float32')
X_cat_test = cv_cat.transform(df_test.category_name)
X_ohe_test = ohe.transform(df_test[['item_condition_id', 'shipping']].values)
X_brand_test = cv_brand.transform(df_test.brand_name)
X_desc_test = cv_desc.transform(df_test.item_description).astype('float32')

X_test = sp.hstack([X_name_test, X_cat_test, X_brand_test, X_desc_test, X_ohe_test], format='csr')

y_pred = model.predict(X_test)
y_pred = np.expm1(y_pred)

df_out = pd.DataFrame()
df_out['test_id'] = df_test.test_id
df_out['price'] = y_pred

df_out.to_csv('submission_ftrl.csv', index=False)
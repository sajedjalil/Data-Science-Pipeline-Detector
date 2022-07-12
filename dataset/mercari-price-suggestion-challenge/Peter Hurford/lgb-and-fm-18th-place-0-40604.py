import time
start_time = time.time()

SUBMIT_MODE = True


import pandas as pd
import numpy as np
import time
import gc
import string
import re

from nltk.corpus import stopwords

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression
from sklearn.preprocessing import LabelBinarizer

import wordbatch
from wordbatch.extractors import WordBag
from wordbatch.models import FM_FTRL

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import MultinomialNB
import lightgbm as lgb


def rmse(predicted, actual):
    return np.sqrt(((predicted - actual) ** 2).mean())


def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


class TargetEncoder:
    # Adapted from https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
    def __repr__(self):
        return 'TargetEncoder'

    def __init__(self, cols, smoothing=1, min_samples_leaf=1, noise_level=0, keep_original=False):
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.keep_original = keep_original

    @staticmethod
    def add_noise(series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))

    def encode(self, train, test, target):
        for col in self.cols:
            if self.keep_original:
                train[col + '_te'], test[col + '_te'] = self.encode_column(train[col], test[col], target)
            else:
                train[col], test[col] = self.encode_column(train[col], test[col], target)
        return train, test

    def encode_column(self, trn_series, tst_series, target):
        temp = pd.concat([trn_series, target], axis=1)
        # Compute target mean
        averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
        # Compute smoothing
        smoothing = 1 / (1 + np.exp(-(averages["count"] - self.min_samples_leaf) / self.smoothing))
        # Apply average function to all target data
        prior = target.mean()
        # The bigger the count the less full_avg is taken into account
        averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(['mean', 'count'], axis=1, inplace=True)
        # Apply averages to trn and tst series
        ft_trn_series = pd.merge(
            trn_series.to_frame(trn_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=trn_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_trn_series.index = trn_series.index
        ft_tst_series = pd.merge(
            tst_series.to_frame(tst_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=tst_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_tst_series.index = tst_series.index
        return self.add_noise(ft_trn_series, self.noise_level), self.add_noise(ft_tst_series, self.noise_level)   


def to_number(x):
    try:
        if not x.isdigit():
            return 0
        x = int(x)
        if x > 100:
            return 100
        else:
            return x
    except:
        return 0

def sum_numbers(desc):
    if not isinstance(desc, str):
        return 0
    try:
        return sum([to_number(s) for s in desc.split()])
    except:
        return 0


# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')
non_alphanumpunct = re.compile(u'[^A-Za-z0-9\.?!,; \(\)\[\]\'\"\$]+')
RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])

def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])

def clean_name(x):
    if len(x):
        x = non_alphanums.sub(' ', x).split()
        if len(x):
            return x[0].lower()
    return ''

    
print('[{}] Finished defining stuff'.format(time.time() - start_time))


train = pd.read_table('../input/train.tsv', engine='c', 
                      dtype={'item_condition_id': 'category',
                             'shipping': 'category',
                            }, 
                     converters={'category_name': split_cat})
test = pd.read_table('../input/test.tsv', engine='c', 
                      dtype={'item_condition_id': 'category',
                             'shipping': 'category',
                            },
                    converters={'category_name': split_cat})
print('[{}] Finished load data'.format(time.time() - start_time))

train['is_train'] = 1
test['is_train'] = 0
print('[{}] Compiled train / test'.format(time.time() - start_time))
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)

train = train[train.price != 0].reset_index(drop=True)
print('[{}] Removed nonzero price'.format(time.time() - start_time))
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)

y = np.log1p(train['price'])
nrow_train = train.shape[0]

merge = pd.concat([train, test])
submission = test[['test_id']]
print('[{}] Compiled merge'.format(time.time() - start_time))
print('Merge shape: ', merge.shape)


del train
del test
merge.drop(['train_id', 'test_id', 'price'], axis=1, inplace=True)
gc.collect()
print('[{}] Garbage collection'.format(time.time() - start_time))


merge['gencat_name'] = merge['category_name'].str.get(0).replace('', 'missing').astype('category')
merge['subcat1_name'] = merge['category_name'].str.get(1).fillna('missing').astype('category')
merge['subcat2_name'] = merge['category_name'].str.get(2).fillna('missing').astype('category')
merge.drop('category_name', axis=1, inplace=True)
print('[{}] Split categories completed.'.format(time.time() - start_time))

merge['item_condition_id'] = merge['item_condition_id'].cat.add_categories(['missing']).fillna('missing')
merge['shipping'] = merge['shipping'].cat.add_categories(['missing']).fillna('missing')
merge['item_description'].fillna('missing', inplace=True)
merge['brand_name'] = merge['brand_name'].fillna('missing').astype('category')
print('[{}] Handle missing completed.'.format(time.time() - start_time))


merge['name_first'] = merge['name'].apply(clean_name)
print('[{}] FE 1/37'.format(time.time() - start_time))
merge['name_first_count'] = merge.groupby('name_first')['name_first'].transform('count')
print('[{}] FE 2/37'.format(time.time() - start_time))
merge['gencat_name_count'] = merge.groupby('gencat_name')['gencat_name'].transform('count')
print('[{}] FE 3/37'.format(time.time() - start_time))
merge['subcat1_name_count'] = merge.groupby('subcat1_name')['subcat1_name'].transform('count')
print('[{}] FE 4/37'.format(time.time() - start_time))
merge['subcat2_name_count'] = merge.groupby('subcat2_name')['subcat2_name'].transform('count')
print('[{}] FE 5/37'.format(time.time() - start_time))
merge['brand_name_count'] = merge.groupby('brand_name')['brand_name'].transform('count')
print('[{}] FE 6/37'.format(time.time() - start_time))
merge['NameLower'] = merge.name.str.count('[a-z]')
print('[{}] FE 7/37'.format(time.time() - start_time))
merge['DescriptionLower'] = merge.item_description.str.count('[a-z]')
print('[{}] FE 8/37'.format(time.time() - start_time))
merge['NameUpper'] = merge.name.str.count('[A-Z]')
print('[{}] FE 9/37'.format(time.time() - start_time))
merge['DescriptionUpper'] = merge.item_description.str.count('[A-Z]')
print('[{}] FE 10/37'.format(time.time() - start_time))
merge['name_len'] = merge['name'].apply(lambda x: len(x))
print('[{}] FE 11/37'.format(time.time() - start_time))
merge['des_len'] = merge['item_description'].apply(lambda x: len(x))
print('[{}] FE 12/37'.format(time.time() - start_time))
merge['name_desc_len_ratio'] = merge['name_len']/merge['des_len']
print('[{}] FE 13/37'.format(time.time() - start_time))
merge['desc_word_count'] = merge['item_description'].apply(lambda x: len(x.split()))
print('[{}] FE 14/37'.format(time.time() - start_time))
merge['mean_des'] = merge['item_description'].apply(lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x)) * 10
print('[{}] FE 15/37'.format(time.time() - start_time))
merge['name_word_count'] = merge['name'].apply(lambda x: len(x.split()))
print('[{}] FE 16/37'.format(time.time() - start_time))
merge['mean_name'] = merge['name'].apply(lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x))  * 10
print('[{}] FE 17/37'.format(time.time() - start_time))
merge['desc_letters_per_word'] = merge['des_len'] / merge['desc_word_count']
print('[{}] FE 18/37'.format(time.time() - start_time))
merge['name_letters_per_word'] = merge['name_len'] / merge['name_word_count']
print('[{}] FE 19/37'.format(time.time() - start_time))
merge['NameLowerRatio'] = merge['NameLower'] / merge['name_len']
print('[{}] FE 20/37'.format(time.time() - start_time))
merge['DescriptionLowerRatio'] = merge['DescriptionLower'] / merge['des_len']
print('[{}] FE 21/37'.format(time.time() - start_time))
merge['NameUpperRatio'] = merge['NameUpper'] / merge['name_len']
print('[{}] FE 22/37'.format(time.time() - start_time))
merge['DescriptionUpperRatio'] = merge['DescriptionUpper'] / merge['des_len']
print('[{}] FE 23/37'.format(time.time() - start_time))
merge['NamePunctCount'] = merge.name.str.count(RE_PUNCTUATION)
print('[{}] FE 24/37'.format(time.time() - start_time))
merge['DescriptionPunctCount'] = merge.item_description.str.count(RE_PUNCTUATION)
print('[{}] FE 25/37'.format(time.time() - start_time))
merge['NamePunctCountRatio'] = merge['NamePunctCount'] / merge['name_word_count']
print('[{}] FE 26/37'.format(time.time() - start_time))
merge['DescriptionPunctCountRatio'] = merge['DescriptionPunctCount'] / merge['desc_word_count']
print('[{}] FE 27/37'.format(time.time() - start_time))
merge['NameDigitCount'] = merge.name.str.count('[0-9]')
print('[{}] FE 28/37'.format(time.time() - start_time))
merge['DescriptionDigitCount'] = merge.item_description.str.count('[0-9]')
print('[{}] FE 29/37'.format(time.time() - start_time))
merge['NameDigitCountRatio'] = merge['NameDigitCount'] / merge['name_word_count']
print('[{}] FE 30/37'.format(time.time() - start_time))
merge['DescriptionDigitCountRatio'] = merge['DescriptionDigitCount']/merge['desc_word_count']
print('[{}] FE 31/37'.format(time.time() - start_time))
merge['stopword_ratio_desc'] = merge['item_description'].apply(lambda x: len([w for w in x.split() if w in stopwords])) / merge['desc_word_count']
print('[{}] FE 32/37'.format(time.time() - start_time))
merge['num_sum'] = merge['item_description'].apply(sum_numbers) 
print('[{}] FE 33/37'.format(time.time() - start_time))
merge['weird_characters_desc'] = merge['item_description'].str.count(non_alphanumpunct)
print('[{}] FE 34/37'.format(time.time() - start_time))
merge['weird_characters_name'] = merge['name'].str.count(non_alphanumpunct)
print('[{}] FE 35/37'.format(time.time() - start_time))
merge['prices_count'] = merge['item_description'].str.count('[rm]')
print('[{}] FE 36/37'.format(time.time() - start_time))
merge['price_in_name'] = merge['item_description'].str.contains('[rm]', regex=False).astype('int')
print('[{}] FE 37/37'.format(time.time() - start_time))

cols = set(merge.columns.values)
basic_cols = {'name', 'item_condition_id', 'brand_name',
  'shipping', 'item_description', 'gencat_name',
  'subcat1_name', 'subcat2_name', 'name_first', 'is_train'}

cols_to_normalize = cols - basic_cols - {'price_in_name'}
other_cols = basic_cols | {'price_in_name'}

merge_to_normalize = merge[list(cols_to_normalize)]
merge_to_normalize = (merge_to_normalize - merge_to_normalize.mean()) / (merge_to_normalize.max() - merge_to_normalize.min())
print('[{}] FE Normalized'.format(time.time() - start_time))

merge = merge[list(other_cols)]
merge = pd.concat([merge, merge_to_normalize],axis=1)
print('[{}] FE Merged'.format(time.time() - start_time))

del(merge_to_normalize)
gc.collect()
print('[{}] Garbage collection'.format(time.time() - start_time))


df_test = merge.loc[merge['is_train'] == 0]
df_train = merge.loc[merge['is_train'] == 1]
del merge
gc.collect()
df_test = df_test.drop(['is_train'], axis=1)
df_train = df_train.drop(['is_train'], axis=1)

if SUBMIT_MODE:
    y_train = y
    del y
    gc.collect()
else:
    df_train, df_test, y_train, y_test = train_test_split(df_train, y, test_size=0.2, random_state=144)

print('[{}] Splitting completed.'.format(time.time() - start_time))


wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
                                                              "hash_ngrams_weights": [1.5, 1.0],
                                                              "hash_size": 2 ** 29,
                                                              "norm": None,
                                                              "tf": 'binary',
                                                              "idf": None,
                                                              }), procs=8)
wb.dictionary_freeze = True
X_name_train = wb.fit_transform(df_train['name'])
X_name_test = wb.transform(df_test['name'])
del(wb)
mask = np.where(X_name_train.getnnz(axis=0) > 3)[0]
X_name_train = X_name_train[:, mask]
X_name_test = X_name_test[:, mask]
print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))


wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
                                                              "hash_ngrams_weights": [1.0, 1.0],
                                                              "hash_size": 2 ** 28,
                                                              "norm": "l2",
                                                              "tf": 1.0,
                                                              "idf": None}), procs=8)
wb.dictionary_freeze = True
X_description_train = wb.fit_transform(df_train['item_description'])
X_description_test = wb.transform(df_test['item_description'])
del(wb)
mask = np.where(X_description_train.getnnz(axis=0) > 3)[0]
X_description_train = X_description_train[:, mask]
X_description_test = X_description_test[:, mask]
print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))


X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_description_train, y_train,
                                                              test_size = 0.5,
                                                              shuffle = False)
print('[{}] Finished splitting'.format(time.time() - start_time))

# Ridge adapted from https://www.kaggle.com/object/more-effective-ridge-script?scriptVersionId=1851819
model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
model.fit(X_train_1, y_train_1)
print('[{}] Finished to train desc ridge (1)'.format(time.time() - start_time))
desc_ridge_preds1 = model.predict(X_train_2)
desc_ridge_preds1f = model.predict(X_description_test)
print('[{}] Finished to predict desc ridge (1)'.format(time.time() - start_time))
model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
model.fit(X_train_2, y_train_2)
print('[{}] Finished to train desc ridge (2)'.format(time.time() - start_time))
desc_ridge_preds2 = model.predict(X_train_1)
desc_ridge_preds2f = model.predict(X_description_test)
print('[{}] Finished to predict desc ridge (2)'.format(time.time() - start_time))
desc_ridge_preds_oof = np.concatenate((desc_ridge_preds2, desc_ridge_preds1), axis=0)
desc_ridge_preds_test = (desc_ridge_preds1f + desc_ridge_preds2f) / 2.0
print('RMSLE OOF: {}'.format(rmse(desc_ridge_preds_oof, y_train)))
if not SUBMIT_MODE:
    print('RMSLE TEST: {}'.format(rmse(desc_ridge_preds_test, y_test)))


X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_name_train, y_train,
                                                              test_size = 0.5,
                                                              shuffle = False)
print('[{}] Finished splitting'.format(time.time() - start_time))

# Ridge adapted from https://www.kaggle.com/object/more-effective-ridge-script?scriptVersionId=1851819
model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
model.fit(X_train_1, y_train_1)
print('[{}] Finished to train name ridge (1)'.format(time.time() - start_time))
name_ridge_preds1 = model.predict(X_train_2)
name_ridge_preds1f = model.predict(X_name_test)
print('[{}] Finished to predict name ridge (1)'.format(time.time() - start_time))
model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
model.fit(X_train_2, y_train_2)
print('[{}] Finished to train name ridge (2)'.format(time.time() - start_time))
name_ridge_preds2 = model.predict(X_train_1)
name_ridge_preds2f = model.predict(X_name_test)
print('[{}] Finished to predict name ridge (2)'.format(time.time() - start_time))
name_ridge_preds_oof = np.concatenate((name_ridge_preds2, name_ridge_preds1), axis=0)
name_ridge_preds_test = (name_ridge_preds1f + name_ridge_preds2f) / 2.0
print('RMSLE OOF: {}'.format(rmse(name_ridge_preds_oof, y_train)))
if not SUBMIT_MODE:
    print('RMSLE TEST: {}'.format(rmse(name_ridge_preds_test, y_test)))


del X_train_1
del X_train_2
del y_train_1
del y_train_2
del name_ridge_preds1
del name_ridge_preds1f
del name_ridge_preds2
del name_ridge_preds2f
del desc_ridge_preds1
del desc_ridge_preds1f
del desc_ridge_preds2
del desc_ridge_preds2f
gc.collect()
print('[{}] Finished garbage collection'.format(time.time() - start_time))


lb = LabelBinarizer(sparse_output=True)
X_brand_train = lb.fit_transform(df_train['brand_name'])
X_brand_test = lb.transform(df_test['brand_name'])
print('[{}] Finished label binarize `brand_name`'.format(time.time() - start_time))

X_cat_train = lb.fit_transform(df_train['gencat_name'])
X_cat_test = lb.transform(df_test['gencat_name'])
X_cat1_train = lb.fit_transform(df_train['subcat1_name'])
X_cat1_test = lb.transform(df_test['subcat1_name'])
X_cat2_train = lb.fit_transform(df_train['subcat2_name'])
X_cat2_test = lb.transform(df_test['subcat2_name'])
print('[{}] Finished label binarize categories'.format(time.time() - start_time))

X_dummies_train = csr_matrix(
    pd.get_dummies(df_train[list(cols - (basic_cols - {'item_condition_id', 'shipping'}))],
                   sparse=True).values)
print('[{}] Create dummies completed - train'.format(time.time() - start_time))

X_dummies_test = csr_matrix(
    pd.get_dummies(df_test[list(cols - (basic_cols - {'item_condition_id', 'shipping'}))],
                   sparse=True).values)
print('[{}] Create dummies completed - test'.format(time.time() - start_time))

sparse_merge_train = hstack((X_dummies_train, X_description_train, X_brand_train, X_cat_train,
                             X_cat1_train, X_cat2_train, X_name_train)).tocsr()
del X_description_train, lb, X_name_train, X_dummies_train
gc.collect()
print('[{}] Create sparse merge train completed'.format(time.time() - start_time))

sparse_merge_test = hstack((X_dummies_test, X_description_test, X_brand_test, X_cat_test,
                             X_cat1_test, X_cat2_test, X_name_test)).tocsr()
del X_description_test, X_name_test, X_dummies_test
gc.collect()
print('[{}] Create sparse merge test completed'.format(time.time() - start_time))


if SUBMIT_MODE:
    iters = 3
else:
    iters = 1
    rounds = 3

model = FM_FTRL(alpha=0.035, beta=0.001, L1=0.00001, L2=0.15, D=sparse_merge_train.shape[1],
                alpha_fm=0.05, L2_fm=0.0, init_fm=0.01,
                D_fm=100, e_noise=0, iters=iters, inv_link="identity", threads=4)

if SUBMIT_MODE:
    model.fit(sparse_merge_train, y_train)
    print('[{}] Train FM completed'.format(time.time() - start_time))
    predsFM = model.predict(sparse_merge_test)
    print('[{}] Predict FM completed'.format(time.time() - start_time))
else:
    for i in range(rounds):
        model.fit(sparse_merge_train, y_train)
        predsFM = model.predict(sparse_merge_test)
        print('[{}] Iteration {}/{} -- RMSLE: {}'.format(time.time() - start_time, i + 1, rounds, rmse(predsFM, y_test)))

del model
gc.collect()
if not SUBMIT_MODE:
    print("FM_FTRL dev RMSLE:", rmse(predsFM, y_test))


fselect = SelectKBest(f_regression, k=48000)
train_features = fselect.fit_transform(sparse_merge_train, y_train)
test_features = fselect.transform(sparse_merge_test)
print('[{}] Select best completed'.format(time.time() - start_time))


del sparse_merge_train
del sparse_merge_test
gc.collect()
print('[{}] Garbage collection'.format(time.time() - start_time))


tv = TfidfVectorizer(max_features=250000,
                     ngram_range=(1, 3),
                     stop_words=None)
X_name_train = tv.fit_transform(df_train['name'])
print('[{}] Finished TFIDF vectorize `name` (1/2)'.format(time.time() - start_time))
X_name_test = tv.transform(df_test['name'])
print('[{}] Finished TFIDF vectorize `name` (2/2)'.format(time.time() - start_time))

tv = TfidfVectorizer(max_features=500000,
                     ngram_range=(1, 3),
                     stop_words=None)
X_description_train = tv.fit_transform(df_train['item_description'])
print('[{}] Finished TFIDF vectorize `item_description` (1/2)'.format(time.time() - start_time))
X_description_test = tv.transform(df_test['item_description'])
print('[{}] Finished TFIDF vectorize `item_description` (2/2)'.format(time.time() - start_time))

X_dummies_train = csr_matrix(
    pd.get_dummies(df_train[['item_condition_id', 'shipping']], sparse=True).values)
X_dummies_test = csr_matrix(
    pd.get_dummies(df_test[['item_condition_id', 'shipping']], sparse=True).values)

sparse_merge_train = hstack((X_description_train, X_brand_train, X_cat_train,
                             X_cat1_train, X_cat2_train, X_name_train)).tocsr()
del X_dummies_train, X_description_train, X_brand_train, X_cat_train
del X_cat1_train, X_cat2_train, X_name_train
gc.collect()
print('[{}] Create sparse merge train completed'.format(time.time() - start_time))

sparse_merge_test = hstack((X_description_test, X_brand_test, X_cat_test,
                            X_cat1_test, X_cat2_test, X_name_test)).tocsr()
del X_dummies_test, X_description_test, X_brand_test, X_cat_test
del X_cat1_test, X_cat2_test, X_name_test
gc.collect()
print('[{}] Create sparse merge test completed'.format(time.time() - start_time))


X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(sparse_merge_train, y_train,
                                                              test_size = 0.5,
                                                              shuffle = False)
print('[{}] Finished splitting'.format(time.time() - start_time))


model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
model.fit(X_train_1, y_train_1)
print('[{}] Finished to train ridge (1)'.format(time.time() - start_time))
ridge_preds1 = model.predict(X_train_2)
ridge_preds1f = model.predict(sparse_merge_test)
print('[{}] Finished to predict ridge (1)'.format(time.time() - start_time))
model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
model.fit(X_train_2, y_train_2)
print('[{}] Finished to train ridge (2)'.format(time.time() - start_time))
ridge_preds2 = model.predict(X_train_1)
ridge_preds2f = model.predict(sparse_merge_test)
print('[{}] Finished to predict ridge (2)'.format(time.time() - start_time))
ridge_preds_oof = np.concatenate((ridge_preds2, ridge_preds1), axis=0)
ridge_preds_test = (ridge_preds1f + ridge_preds2f) / 2.0
print('RMSLE OOF: {}'.format(rmse(ridge_preds_oof, y_train)))
if not SUBMIT_MODE:
    print('RMSLE TEST: {}'.format(rmse(ridge_preds_test, y_test)))


model = MultinomialNB(alpha=0.01)
model.fit(X_train_1, y_train_1 >= 4)
print('[{}] Finished to train MNB (1)'.format(time.time() - start_time))
mnb_preds1 = model.predict_proba(X_train_2)[:, 1]
mnb_preds1f = model.predict_proba(sparse_merge_test)[:, 1]
print('[{}] Finished to predict MNB (1)'.format(time.time() - start_time))
model = MultinomialNB(alpha=0.01)
model.fit(X_train_2, y_train_2 >= 4)
print('[{}] Finished to train MNB (2)'.format(time.time() - start_time))
mnb_preds2 = model.predict_proba(X_train_1)[:, 1]
mnb_preds2f = model.predict_proba(sparse_merge_test)[:, 1]
print('[{}] Finished to predict MNB (2)'.format(time.time() - start_time))
mnb_preds_oof = np.concatenate((mnb_preds2, mnb_preds1), axis=0)
mnb_preds_test = (mnb_preds1f + mnb_preds2f) / 2.0


del ridge_preds1
del ridge_preds1f
del ridge_preds2
del ridge_preds2f
del mnb_preds1
del mnb_preds1f
del mnb_preds2
del mnb_preds2f
del X_train_1
del X_train_2
del y_train_1
del y_train_2
del sparse_merge_train
del sparse_merge_test
del model
gc.collect()
print('[{}] Finished garbage collection'.format(time.time() - start_time))


df_train['ridge'] = ridge_preds_oof
df_train['name_ridge'] = name_ridge_preds_oof
df_train['desc_ridge'] = desc_ridge_preds_oof
df_train['mnb'] = mnb_preds_oof
df_test['ridge'] = ridge_preds_test
df_test['name_ridge'] = name_ridge_preds_test
df_test['desc_ridge'] = desc_ridge_preds_test
df_test['mnb'] = mnb_preds_test
print('[{}] Finished adding submodels'.format(time.time() - start_time))


f_cats = ['brand_name', 'gencat_name', 'subcat1_name', 'subcat2_name', 'name_first']
target_encode = TargetEncoder(min_samples_leaf=100, smoothing=10, noise_level=0.01,
                              keep_original=True, cols=f_cats)
df_train, df_test = target_encode.encode(df_train, df_test, y_train)
print('[{}] Finished target encoding'.format(time.time() - start_time))


df_train.drop(f_cats, axis=1, inplace=True)
df_test.drop(f_cats, axis=1, inplace=True)
del mnb_preds_oof
del mnb_preds_test
del ridge_preds_oof
del ridge_preds_test
gc.collect()
print('[{}] Finished garbage collection'.format(time.time() - start_time))


cols = ['gencat_name_te', 'brand_name_te', 'subcat1_name_te', 'subcat2_name_te',
        'name_first_te', 'mnb', 'desc_ridge', 'name_ridge', 'ridge']
train_dummies = csr_matrix(df_train[cols].values)
print('[{}] Finished dummyizing model 1/5'.format(time.time() - start_time))
test_dummies = csr_matrix(df_test[cols].values)
print('[{}] Finished dummyizing model 2/5'.format(time.time() - start_time))
del df_train
del df_test
gc.collect()
print('[{}] Finished dummyizing model 3/5'.format(time.time() - start_time))
train_features = hstack((train_features, train_dummies)).tocsr()
print('[{}] Finished dummyizing model 4/5'.format(time.time() - start_time))
test_features = hstack((test_features, test_dummies)).tocsr()
print('[{}] Finished dummyizing model 5/5'.format(time.time() - start_time))


d_train = lgb.Dataset(train_features, label=y_train)
del train_features; gc.collect()
if SUBMIT_MODE:
    watchlist = [d_train]
else:
    d_valid = lgb.Dataset(test_features, label=y_test)
    watchlist = [d_train, d_valid]

params = {
    'learning_rate': 0.15,
    'application': 'regression',
    'max_depth': 13,
    'num_leaves': 400,
    'verbosity': -1,
    'metric': 'RMSE',
    'data_random_seed': 1,
    'bagging_fraction': 0.8,
    'feature_fraction': 0.6,
    'nthread': 4,
    'lambda_l1': 10,
    'lambda_l2': 10
}
print('[{}] Finished compiling LGB'.format(time.time() - start_time))

modelL = lgb.train(params,
                  train_set=d_train,
                  num_boost_round=1350,
                  valid_sets=watchlist,
                  verbose_eval=50)

predsL = modelL.predict(test_features)
predsL[predsL < 0] = 0

if not SUBMIT_MODE:
    print("LGB RMSLE:", rmse(predsL, y_test))

del d_train
del modelL
if not SUBMIT_MODE:
    del d_valid
gc.collect()


preds_final = predsFM * 0.33 + predsL * 0.67
if not SUBMIT_MODE:
    print('Final RMSE: ', rmse(preds_final, y_test))


if SUBMIT_MODE:
    preds_final = np.expm1(preds_final)
    submission['price'] = preds_final
    submission.to_csv('lgb_and_fm_separate_train_test.csv', index=False)
    print('[{}] Writing submission done'.format(time.time() - start_time))
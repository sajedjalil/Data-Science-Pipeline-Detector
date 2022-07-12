import re
import multiprocessing as mp
from time import time
import os
import gc
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
import scipy.sparse as csr_matrix

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import mean_squared_log_error
from sklearn.base import BaseEstimator, TransformerMixin

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K

import wordbatch
from wordbatch.extractors import WordSeq

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['JOBLIB_START_METHOD'] = 'forkserver'


# Define function to calculate root mean square log error.
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(y_pred)))

print("Loading data from files...")
train_df = pd.read_table('../input/train.tsv',
                         engine='c',
                         dtype={'item_condition_id': 'category',
                                'shipping': 'category'},
                         #nrows=10000,
                        )
test_df = pd.read_table('../input/test.tsv',
                         engine='c',
                         dtype={'item_condition_id': 'category',
                                'shipping': 'category'},
                         #nrows=1000,
                        )
print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)

#########################################
########## Preprocessing data ###########
#########################################

class SymSpell:
    def __init__(self, max_edit_distance=3, verbose=0):
        self.max_edit_distance = max_edit_distance
        self.verbose = verbose
        # 0: top suggestion
        # 1: all suggestions of smallest edit distance
        # 2: all suggestions <= max_edit_distance (slower, no early termination)

        self.dictionary = {}
        self.longest_word_length = 0

    def get_deletes_list(self, w):
        """given a word, derive strings with up to max_edit_distance characters
           deleted"""

        deletes = []
        queue = [w]
        for d in range(self.max_edit_distance):
            temp_queue = []
            for word in queue:
                if len(word) > 1:
                    for c in range(len(word)):  # character index
                        word_minus_c = word[:c] + word[c + 1:]
                        if word_minus_c not in deletes:
                            deletes.append(word_minus_c)
                        if word_minus_c not in temp_queue:
                            temp_queue.append(word_minus_c)
            queue = temp_queue

        return deletes

    def create_dictionary_entry(self, w):
        '''add word and its derived deletions to dictionary'''
        # check if word is already in dictionary
        # dictionary entries are in the form: (list of suggested corrections,
        # frequency of word in corpus)
        new_real_word_added = False
        if w in self.dictionary:
            # increment count of word in corpus
            self.dictionary[w] = (self.dictionary[w][0], self.dictionary[w][1] + 1)
        else:
            self.dictionary[w] = ([], 1)
            self.longest_word_length = max(self.longest_word_length, len(w))

        if self.dictionary[w][1] == 1:
            # first appearance of word in corpus
            # n.b. word may already be in dictionary as a derived word
            # (deleting character from a real word)
            # but counter of frequency of word in corpus is not incremented
            # in those cases)
            new_real_word_added = True
            deletes = self.get_deletes_list(w)
            for item in deletes:
                if item in self.dictionary:
                    # add (correct) word to delete's suggested correction list
                    self.dictionary[item][0].append(w)
                else:
                    # note frequency of word in corpus is not incremented
                    self.dictionary[item] = ([w], 0)

        return new_real_word_added

    def create_dictionary_from_arr(self, arr, token_pattern=r'[a-z]+'):
        total_word_count = 0
        unique_word_count = 0

        for line in arr:
            # separate by words by non-alphabetical characters
            words = re.findall(token_pattern, line.lower())
            for word in words:
                total_word_count += 1
                if self.create_dictionary_entry(word):
                    unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def create_dictionary(self, fname):
        total_word_count = 0
        unique_word_count = 0

        with open(fname) as file:
            for line in file:
                # separate by words by non-alphabetical characters
                words = re.findall('[a-z]+', line.lower())
                for word in words:
                    total_word_count += 1
                    if self.create_dictionary_entry(word):
                        unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def get_suggestions(self, string, silent=False):
        """return list of suggested corrections for potentially incorrectly
           spelled word"""
        if (len(string) - self.longest_word_length) > self.max_edit_distance:
            if not silent:
                print("no items in dictionary within maximum edit distance")
            return []

        suggest_dict = {}
        min_suggest_len = float('inf')

        queue = [string]
        q_dictionary = {}  # items other than string that we've checked

        while len(queue) > 0:
            q_item = queue[0]  # pop
            queue = queue[1:]

            # early exit
            if ((self.verbose < 2) and (len(suggest_dict) > 0) and
                    ((len(string) - len(q_item)) > min_suggest_len)):
                break

            # process queue item
            if (q_item in self.dictionary) and (q_item not in suggest_dict):
                if self.dictionary[q_item][1] > 0:
                    # word is in dictionary, and is a word from the corpus, and
                    # not already in suggestion list so add to suggestion
                    # dictionary, indexed by the word with value (frequency in
                    # corpus, edit distance)
                    # note q_items that are not the input string are shorter
                    # than input string since only deletes are added (unless
                    # manual dictionary corrections are added)
                    assert len(string) >= len(q_item)
                    suggest_dict[q_item] = (self.dictionary[q_item][1],
                                            len(string) - len(q_item))
                    # early exit
                    if (self.verbose < 2) and (len(string) == len(q_item)):
                        break
                    elif (len(string) - len(q_item)) < min_suggest_len:
                        min_suggest_len = len(string) - len(q_item)

                # the suggested corrections for q_item as stored in
                # dictionary (whether or not q_item itself is a valid word
                # or merely a delete) can be valid corrections
                for sc_item in self.dictionary[q_item][0]:
                    if sc_item not in suggest_dict:

                        # compute edit distance
                        # suggested items should always be longer
                        # (unless manual corrections are added)
                        assert len(sc_item) > len(q_item)

                        # q_items that are not input should be shorter
                        # than original string
                        # (unless manual corrections added)
                        assert len(q_item) <= len(string)

                        if len(q_item) == len(string):
                            assert q_item == string
                            item_dist = len(sc_item) - len(q_item)

                        # item in suggestions list should not be the same as
                        # the string itself
                        assert sc_item != string

                        # calculate edit distance using, for example,
                        # Damerau-Levenshtein distance
                        item_dist = dameraulevenshtein(sc_item, string)

                        # do not add words with greater edit distance if
                        # verbose setting not on
                        if (self.verbose < 2) and (item_dist > min_suggest_len):
                            pass
                        elif item_dist <= self.max_edit_distance:
                            assert sc_item in self.dictionary  # should already be in dictionary if in suggestion list
                            suggest_dict[sc_item] = (self.dictionary[sc_item][1], item_dist)
                            if item_dist < min_suggest_len:
                                min_suggest_len = item_dist

                        # depending on order words are processed, some words
                        # with different edit distances may be entered into
                        # suggestions; trim suggestion dictionary if verbose
                        # setting not on
                        if self.verbose < 2:
                            suggest_dict = {k: v for k, v in suggest_dict.items() if v[1] <= min_suggest_len}

            # now generate deletes (e.g. a substring of string or of a delete)
            # from the queue item
            # as additional items to check -- add to end of queue
            assert len(string) >= len(q_item)

            # do not add words with greater edit distance if verbose setting
            # is not on
            if (self.verbose < 2) and ((len(string) - len(q_item)) > min_suggest_len):
                pass
            elif (len(string) - len(q_item)) < self.max_edit_distance and len(q_item) > 1:
                for c in range(len(q_item)):  # character index
                    word_minus_c = q_item[:c] + q_item[c + 1:]
                    if word_minus_c not in q_dictionary:
                        queue.append(word_minus_c)
                        q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this

        # queue is now empty: convert suggestions in dictionary to
        # list for output
        if not silent and self.verbose != 0:
            print("number of possible corrections: %i" % len(suggest_dict))
            print("  edit distance for deletions: %i" % self.max_edit_distance)

        # output option 1
        # sort results by ascending order of edit distance and descending
        # order of frequency
        #     and return list of suggested word corrections only:
        # return sorted(suggest_dict, key = lambda x:
        #               (suggest_dict[x][1], -suggest_dict[x][0]))

        # output option 2
        # return list of suggestions with (correction,
        #                                  (frequency in corpus, edit distance)):
        as_list = suggest_dict.items()
        # outlist = sorted(as_list, key=lambda (term, (freq, dist)): (dist, -freq))
        outlist = sorted(as_list, key=lambda x: (x[1][1], -x[1][0]))

        if self.verbose == 0:
            return outlist[0]
        else:
            return outlist

        '''
        Option 1:
        ['file', 'five', 'fire', 'fine', ...]

        Option 2:
        [('file', (5, 0)),
         ('five', (67, 1)),
         ('fire', (54, 1)),
         ('fine', (17, 1))...]  
        '''

    def best_word(self, s, silent=False):
        try:
            return self.get_suggestions(s, silent)[0]
        except:
            return None

def split_cat(text):
    try:
        cats = text.split("/")
        return cats[0], cats[1], cats[2], cats[0] + '/' + cats[1]
    except:
        print("no category")
        return 'other', 'other', 'other', 'other/other'

def brands_filling(dataset):
    vc = dataset['brand_name'].value_counts()
    brands = vc[vc > 0].index
    brand_word = r"[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+"

    many_w_brands = brands[brands.str.contains(' ')]
    one_w_brands = brands[~brands.str.contains(' ')]

    ss2 = SymSpell(max_edit_distance=0)
    ss2.create_dictionary_from_arr(many_w_brands, token_pattern=r'.+')

    ss1 = SymSpell(max_edit_distance=0)
    ss1.create_dictionary_from_arr(one_w_brands, token_pattern=r'.+')

    two_words_re = re.compile(r"(?=(\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+))")

    def find_in_str_ss2(row):
        for doc_word in two_words_re.finditer(row):
            print(doc_word)
            suggestion = ss2.best_word(doc_word.group(1), silent=True)
            if suggestion is not None:
                return doc_word.group(1)
        return ''

    def find_in_list_ss1(list):
        for doc_word in list:
            suggestion = ss1.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    def find_in_list_ss2(list):
        for doc_word in list:
            suggestion = ss2.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    print(f"Before empty brand_name: {len(dataset[dataset['brand_name'] == ''].index)}")

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_name]

    n_desc = dataset[dataset['brand_name'] == '']['item_description'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_desc]

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in n_name]

    desc_lower = dataset[dataset['brand_name'] == '']['item_description'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in desc_lower]

    print(f"After empty brand_name: {len(dataset[dataset['brand_name'] == ''].index)}")

    del ss1, ss2
    gc.collect()

def preprocess_regex(dataset):
    karats_regex = r'(\d)([\s-]?)(karat|karats|carat|carats|kt)([^\w])'
    karats_repl = r'\1k\4'

    unit_regex = r'(\d+)[\s-]([a-z]{2})(\s)'
    unit_repl = r'\1\2\3'

    dataset['name'] = dataset['name'].str.replace(karats_regex, karats_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(karats_regex, karats_repl)

    dataset['name'] = dataset['name'].str.replace(unit_regex, unit_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(unit_regex, unit_repl)

def preprocess_pandas(train, test):
    print("Droping training example with non-positive price...")
    train = train[train.price > 0.0].reset_index(drop=True)
    print('Train shape without non-positive price: ', train.shape)

    train, dev = train_test_split(train, random_state=123, test_size=0.01)    
    y_train = np.log1p(train.price).values.reshape(-1, 1)
    y_dev = np.log1p(dev.price).values.reshape(-1, 1)

    merge = pd.concat([train, dev, test])

    del train, dev, test
    gc.collect()

    print("Processing categorical data and handling missing data...")
    merge['has_category'] = (merge['category_name'].notnull()).astype('category')

    merge['category_name'] = merge['category_name'].fillna('other/other/other').str.lower()  
    merge['general_cat'], merge['subcat_1'], merge['subcat_2'], merge['gen_subcat1'] = \
        zip(*merge['category_name'].apply(lambda x: split_cat(x)))

    merge['has_brand'] = (merge['brand_name'].notnull()).astype('category')

    merge['gencat_cond'] = merge['general_cat'].map(str) + '_' + merge['item_condition_id'].astype(str)
    merge['subcat_1_cond'] = merge['subcat_1'].map(str) + '_' + merge['item_condition_id'].astype(str)
    merge['subcat_2_cond'] = merge['subcat_2'].map(str) + '_' + merge['item_condition_id'].astype(str)

    merge['name'] = merge['name'].fillna('').str.lower()
    merge['brand_name'] = merge['brand_name'].fillna('').str.lower()
    merge['item_description'] = merge['item_description'].fillna('').str.lower() \
        .replace(to_replace='No description yet', value='')

    print("Normalizing data for text fields...")
    preprocess_regex(merge)

    print("Processing brand_name...")
    brands_filling(merge)

    merge['name'] = merge['name'] + ' ' + merge['brand_name']

    merge['item_description'] = merge['item_description'] \
                                + ' ' + merge['name'] \
                                + ' ' + merge['subcat_1'] \
                                + ' ' + merge['subcat_2'] \
                                + ' ' + merge['general_cat'] \
                                + ' ' + merge['brand_name']

    merge.drop(['price', 'test_id', 'train_id'], axis=1, inplace=True)

    return merge, y_train, y_dev


print("Proprecessing data...")
submission = pd.DataFrame({"test_id": test_df.test_id})
full_df, y_train, y_dev = preprocess_pandas(train_df, test_df)

# Calculate number of train/dev/test examples.
n_trains = y_train.shape[0]
n_devs = y_dev.shape[0]

#######################################################
################### Ridge Model #######################
#######################################################

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        dt = dataframe[self.field].dtype
        if is_categorical_dtype(dt):
            return dataframe[self.field].cat.codes[:, None]
        elif is_numeric_dtype(dt):
            return dataframe[self.field][:, None]
        else:
            return dataframe[self.field]

class DropColumnsByDf(BaseEstimator, TransformerMixin):
    def __init__(self, min_df=1, max_df=1.0):
        self.min_df = min_df
        self.max_df = max_df

    def fit(self, X, y=None):
        m = X.tocsc()
        self.nnz_cols = ((m != 0).sum(axis=0) >= self.min_df).A1
        if self.max_df < 1.0:
            max_df = m.shape[0] * self.max_df
            self.nnz_cols = self.nnz_cols & ((m != 0).sum(axis=0) <= max_df).A1
        return self

    def transform(self, X, y=None):
        m = X.tocsc()
        return m[:, self.nnz_cols]

meta_params = {'name_ngram': (1, 2),
               'name_max_f': 75000,
               'name_min_df': 10,

               'category_ngram': (2, 3),
               'category_token': '.+',
               'category_min_df': 10,

               'brand_min_df': 10,

               'desc_ngram': (1, 3),
               'desc_max_f': 150000,
               'desc_max_df': 0.5,
               'desc_min_df': 10}

stopwords = frozenset(['the', 'a', 'an', 'is', 'it', 'this', ])

vectorizer = FeatureUnion([
    ('name', Pipeline([
        ('select', ItemSelector('name')),
        ('transform', HashingVectorizer(
            ngram_range=(1, 2),
            n_features=2 ** 27,
            norm='l2',
            lowercase=False,
            stop_words=stopwords
        )),
        ('drop_cols', DropColumnsByDf(min_df=2))
    ])),
    ('category_name', Pipeline([
        ('select', ItemSelector('category_name')),
        ('transform', HashingVectorizer(
            ngram_range=(1, 1),
            token_pattern='.+',
            tokenizer=split_cat,
            n_features=2 ** 27,
            norm='l2',
            lowercase=False
        )),
        ('drop_cols', DropColumnsByDf(min_df=2))
    ])),
    ('brand_name', Pipeline([
        ('select', ItemSelector('brand_name')),
        ('transform', CountVectorizer(
            token_pattern='.+',
            min_df=2,
            lowercase=False
        )),
    ])),
    ('gencat_cond', Pipeline([
        ('select', ItemSelector('gencat_cond')),
        ('transform', CountVectorizer(
            token_pattern='.+',
            min_df=2,
            lowercase=False
        )),
    ])),
    ('subcat_1_cond', Pipeline([
        ('select', ItemSelector('subcat_1_cond')),
        ('transform', CountVectorizer(
            token_pattern='.+',
            min_df=2,
            lowercase=False
        )),
    ])),
    ('subcat_2_cond', Pipeline([
        ('select', ItemSelector('subcat_2_cond')),
        ('transform', CountVectorizer(
            token_pattern='.+',
            min_df=2,
            lowercase=False
        )),
    ])),
    ('has_brand', Pipeline([
        ('select', ItemSelector('has_brand')),
        ('ohe', OneHotEncoder())
    ])),
    ('shipping', Pipeline([
        ('select', ItemSelector('shipping')),
        ('ohe', OneHotEncoder())
    ])),
    ('item_condition_id', Pipeline([
        ('select', ItemSelector('item_condition_id')),
        ('ohe', OneHotEncoder())
    ])),
    ('item_description', Pipeline([
        ('select', ItemSelector('item_description')),
        ('hash', HashingVectorizer(
            ngram_range=(1, 3),
            n_features=2 ** 27,
            dtype=np.float32,
            norm='l2',
            lowercase=False,
            stop_words=stopwords
        )),
        ('drop_cols', DropColumnsByDf(min_df=2)),
    ]))
], n_jobs=1)

print("Vecterizing data...")
sparse = vectorizer.fit_transform(full_df)

tfidf_transformer = TfidfTransformer()
X_full = tfidf_transformer.fit_transform(sparse)

X_train = X_full[:n_trains]
X_dev = X_full[n_trains:n_trains+n_devs]
X_test = X_full[n_trains+n_devs:]

print(X_full.shape, X_train.shape, X_dev.shape, X_test.shape)

del sparse, X_full, vectorizer, tfidf_transformer
gc.collect()

def intersect_drop_columns(train: csr_matrix, dev: csr_matrix, valid: csr_matrix, min_df=0):
    t = train.tocsc()
    d = dev.tocsc()
    v = valid.tocsc()
    
    nnz_train = ((t != 0).sum(axis=0) >= min_df).A1
    nnz_dev = ((d != 0).sum(axis=0) >= min_df).A1
    nnz_valid = ((v != 0).sum(axis=0) >= min_df).A1
    nnz_cols = (nnz_train | nnz_dev) & nnz_valid
    res = t[:, nnz_cols], d[:, nnz_cols], v[:, nnz_cols]
    return res

print("Dropping columns that present only on train or test dataset...")
X_train, X_dev, X_test = intersect_drop_columns(X_train, X_dev, X_test, min_df=1)
print(X_train.shape, X_dev.shape, X_test.shape)

print("Training Ridge model...")
ridge = Ridge(solver='auto', fit_intercept=True, alpha=0.4, max_iter=200, normalize=False, tol=0.01)
ridge.fit(X_train, y_train)

# Evaluating Ridge model
y_dev_preds_ridge = ridge.predict(X_dev)
y_dev_preds_ridge = y_dev_preds_ridge.reshape(-1, 1)
print("RMSL error on dev set:", rmsle(y_dev, y_dev_preds_ridge))

print("Predicting prices for test dataset...")
y_pred_ridge = ridge.predict(X_test)
y_pred_ridge[y_pred_ridge < 0.0] = 0.0
y_pred_redge = y_pred_ridge.reshape(-1, 1)

del ridge, X_train, X_dev, X_test
gc.collect()

########################################################
##################### RNN Model ########################
########################################################

print("Processing categorical data...")

full_df.shipping = full_df.shipping.astype(int)
full_df.item_condition_id = full_df.item_condition_id.astype(int)

le = LabelEncoder()

full_df.category_name = le.fit_transform(full_df.category_name)
full_df.brand_name = le.fit_transform(full_df.brand_name)

del le

def normalize_text(text):
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    text = text.lower()
    return text.translate(str.maketrans(filters, " " * len(filters)))

print("Transforming texts to sequences...")
wb = wordbatch.WordBatch(normalize_text=normalize_text, extractor=(WordSeq, {}), procs=8)
wb.fit(np.hstack([full_df.name, full_df.item_description]))

full_df["seq_name"] = wb.transform(full_df.name)
full_df["seq_item_description"] = wb.transform(full_df.item_description)

del wb

MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75
MAX_TEXT = np.max([
    np.max(full_df.seq_name.apply(lambda x: max(x))),
    np.max(full_df.seq_item_description.apply(lambda x: max(x))),
]) + 1
MAX_CATEGORY = np.max(full_df.category_name.max()) + 1
MAX_BRAND = np.max(full_df.brand_name.max()) + 1
MAX_CONDITION = np.max(full_df.item_condition_id.max()) + 1

def get_keras_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        'brand_name': dataset.brand_name.values,
        'category_name': dataset.category_name.values,
        'item_condition': dataset.item_condition_id.values,
        'num_vars': dataset.shipping.values.reshape(-1, 1),
    }
    return X

print("Getting data for RNN model...")
train = full_df[:n_trains]
dev = full_df[n_trains:n_trains+n_devs]
test = full_df[n_trains+n_devs:]

X_train = get_keras_data(train)
X_dev = get_keras_data(dev)
X_test = get_keras_data(test)

# Define function for constructing RNN model.
def new_rnn_model(lr=0.001, decay=0.0):    
    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name='category_name')
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

    # Embeddings layers
    emb_name = Embedding(MAX_TEXT, 20)(name)
    emb_item_desc = Embedding(MAX_TEXT, 50)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 5)(brand_name)
    emb_category_name = Embedding(MAX_CATEGORY, 5)(category_name)

    # rnn layers
    rnn_layer1 = GRU(16) (emb_item_desc)
    rnn_layer2 = GRU(8) (emb_name)

    # main layers
    main_l = concatenate([
        Flatten() (emb_brand_name),
        Flatten() (emb_category_name),
        item_condition,
        rnn_layer1,
        rnn_layer2,
        num_vars,
    ])

    main_l = Dense(256)(main_l)
    main_l = Activation('elu')(main_l)

    main_l = Dense(128)(main_l)
    main_l = Activation('elu')(main_l)

    main_l = Dense(64)(main_l)
    main_l = Activation('elu')(main_l)

    # the output layer.
    output = Dense(1, activation="linear") (main_l)

    model = Model([name, item_desc, brand_name , category_name, item_condition, num_vars], output)

    optimizer = Adam(lr=lr, decay=decay)
    model.compile(loss="mse", optimizer=optimizer)

    return model

print("Training RNN model...")
# Set hyper parameters for the model.
BATCH_SIZE = 1024
epochs = 2

# Calculate learning rate decay.
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(n_trains / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.007, 0.0005
lr_decay = exp_decay(lr_init, lr_fin, steps)

# Create model and fit it with training dataset.
rnn = new_rnn_model(lr=lr_init, decay=lr_decay)
rnn.fit(
        X_train, y_train, epochs=epochs, batch_size=BATCH_SIZE,
        validation_data=(X_dev, y_dev), verbose=2,
)

y_dev_preds_rnn = rnn.predict(X_dev, batch_size=BATCH_SIZE)
print("RMSL error on dev set:", rmsle(y_dev, y_dev_preds_rnn))

print("Predicting prices for test dataset...")
y_pred_rnn = rnn.predict(X_test, batch_size=BATCH_SIZE)
y_pred_rnn[y_pred_rnn < 0.0] = 0.0
y_pred_rnn = y_pred_rnn.reshape(-1, 1)

# Clean-up unused data to get back memory.
del train, dev, test, X_train, X_dev, X_test, full_df, rnn
gc.collect()

print("Evaluating RNN + Ridge model...")
def aggregate_predicts(Y1, Y2):
    assert Y1.shape == Y2.shape
    ratio = 0.42
    return Y1 * ratio + Y2 * (1.0 - ratio)

y_dev_preds = aggregate_predicts(y_dev_preds_rnn, y_dev_preds_ridge)
print("RMSL error for RNN + Ridge on dev dataset:", rmsle(y_dev, y_dev_preds))

print("Creating submission...")
y_pred = aggregate_predicts(y_pred_rnn, y_pred_ridge)
submission["price"] = np.expm1(y_pred.reshape(-1))
submission.to_csv("submission.csv", index=False)
import multiprocessing as mp
import pandas as pd
from time import time
from scipy.sparse import csr_matrix
import os
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import gc
from sklearn.base import BaseEstimator, TransformerMixin
import re
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from sklearn.model_selection import train_test_split

develop = False
# develop= True
train_file = '../input/train.tsv'
test_file = '../input/test_stg2.tsv'
random_seed = np.random.randint(1000)

def dameraulevenshtein(seq1, seq2):
    """Calculate the Damerau-Levenshtein distance between sequences.

    This method has not been modified from the original.
    Source: http://mwh.geek.nz/2009/04/26/python-damerau-levenshtein-distance/

    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.

    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.

    This implementation is O(N*M) time and O(M) space, for N and M the
    lengths of the two sequences.

    >>> dameraulevenshtein('ba', 'abc')
    2
    >>> dameraulevenshtein('fee', 'deed')
    2

    It works with arbitrary sequences too:
    >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
    2
    """
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = (oneago, thisrow, [0] * len(seq2) + [x + 1])
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]


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

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field, start_time=time()):
        self.field = field
        self.start_time = start_time

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        print('[{}] select {}'.format(time()-self.start_time, self.field))
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


def get_rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))


def split_cat(text):
    try:
        cats = text.split("/")
        return cats[0], cats[1], cats[2], cats[0] + '/' + cats[1]
    except:
        print("no category")
        return 'other', 'other', 'other', 'other/other'


def brands_filling(dataset, train = True, ss1 = None, ss2 = None):
    if train:
        vc = dataset['brand_name'].value_counts()
        brands = vc[vc > 0].index

        many_w_brands = brands[brands.str.contains(' ')]
        one_w_brands = brands[~brands.str.contains(' ')]

        ss2 = SymSpell(max_edit_distance=0)
        ss2.create_dictionary_from_arr(many_w_brands, token_pattern=r'.+')

        ss1 = SymSpell(max_edit_distance=0)
        ss1.create_dictionary_from_arr(one_w_brands, token_pattern=r'.+')

    two_words_re = re.compile(r"(?=(\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+))")
    brand_word = r"[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+"

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

    print("Before empty brand_name: {}".format(len(dataset[dataset['brand_name'] == ''].index)))

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

    print("After empty brand_name: {}".format(len(dataset[dataset['brand_name'] == ''].index)))
    return(ss1, ss2)

def preprocess_regex(dataset, start_time=time()):
    karats_regex = r'(\d)([\s-]?)(karat|karats|carat|carats|kt)([^\w])'
    karats_repl = r'\1k\4'

    unit_regex = r'(\d+)[\s-]([a-z]{2})(\s)'
    unit_repl = r'\1\2\3'

    dataset['name'] = dataset['name'].str.replace(karats_regex, karats_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(karats_regex, karats_repl)
    print('[{}] Karats normalized.'.format(time() - start_time))

    dataset['name'] = dataset['name'].str.replace(unit_regex, unit_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(unit_regex, unit_repl)
    print('[{}] Units glued.'.format(time() - start_time))


def preprocess_pandas(dataset, start_time, train = True, ss1 = None, ss2 = None):
    print('dataset shape without zero price: ', dataset.shape)

    dataset['has_category'] = (dataset['category_name'].notnull()).astype('category')
    print('[{}] Has_category filled.'.format(time() - start_time))

    dataset['category_name'] = dataset['category_name'] \
        .fillna('other/other/other') \
        .str.lower() \
        .astype(str)
    dataset['general_cat'], dataset['subcat_1'], dataset['subcat_2'], dataset['gen_subcat1'] = \
        zip(*dataset['category_name'].apply(lambda x: split_cat(x)))
    print('[{}] Split categories completed.'.format(time() - start_time))

    dataset['has_brand'] = (dataset['brand_name'].notnull()).astype('category')
    print('[{}] Has_brand filled.'.format(time() - start_time))

    dataset['gencat_cond'] = dataset['general_cat'].map(str) + '_' + dataset['item_condition_id'].astype(str)
    dataset['subcat_1_cond'] = dataset['subcat_1'].map(str) + '_' + dataset['item_condition_id'].astype(str)
    dataset['subcat_2_cond'] = dataset['subcat_2'].map(str) + '_' + dataset['item_condition_id'].astype(str)
    print('[{}] Categories and item_condition_id concancenated.'.format(time() - start_time))

    dataset['name'] = dataset['name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    dataset['brand_name'] = dataset['brand_name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    dataset['item_description'] = dataset['item_description'] \
        .fillna('') \
        .str.lower() \
        .replace(to_replace='No description yet', value='')
    print('[{}] Missing filled.'.format(time() - start_time))

    preprocess_regex(dataset, start_time)
    if train:
        ss1, ss2 = brands_filling(dataset)
    else:
        ss1, ss2 = brands_filling(dataset, train = train, ss1 = ss1, ss2 = ss2)
    print('[{}] Brand name filled.'.format(time() - start_time))

    dataset['name'] = dataset['name'] + ' ' + dataset['brand_name']
    print('[{}] Name concancenated.'.format(time() - start_time))

    dataset['item_description'] = dataset['item_description'] \
                                + ' ' + dataset['name'] \
                                + ' ' + dataset['subcat_1'] \
                                + ' ' + dataset['subcat_2'] \
                                + ' ' + dataset['general_cat'] \
                                + ' ' + dataset['brand_name']
    print('[{}] Item description concatenated.'.format(time() - start_time))
    return(dataset, ss1, ss2)
    
start_time = time()

train = pd.read_table(train_file,
                      engine='c',
                      dtype={'item_condition_id': 'category',
                             'shipping': 'category'}
                      )
train = train[train.price > 0.0].reset_index(drop=True)
if develop:
    train, dev = train_test_split(train, test_size=0.05, random_state = random_seed)
    dev_y = np.log1p(train["price"])
train_y = np.log1p(train["price"])
train, ss1, ss2 = preprocess_pandas(train, start_time)

stopwords = frozenset(['the', 'a', 'an', 'is', 'it', 'this', ])
vectorizer = FeatureUnion([
    ('name', Pipeline([
        ('select', ItemSelector('name', start_time=start_time)),
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
        ('select', ItemSelector('category_name', start_time=start_time)),
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
        ('select', ItemSelector('brand_name', start_time=start_time)),
        ('transform', CountVectorizer(
            token_pattern='.+',
            min_df=2,
            lowercase=False
        )),
    ])),
    ('gencat_cond', Pipeline([
        ('select', ItemSelector('gencat_cond', start_time=start_time)),
        ('transform', CountVectorizer(
            token_pattern='.+',
            min_df=2,
            lowercase=False
        )),
    ])),
    ('subcat_1_cond', Pipeline([
        ('select', ItemSelector('subcat_1_cond', start_time=start_time)),
        ('transform', CountVectorizer(
            token_pattern='.+',
            min_df=2,
            lowercase=False
        )),
    ])),
    ('subcat_2_cond', Pipeline([
        ('select', ItemSelector('subcat_2_cond', start_time=start_time)),
        ('transform', CountVectorizer(
            token_pattern='.+',
            min_df=2,
            lowercase=False
        )),
    ])),
    ('has_brand', Pipeline([
        ('select', ItemSelector('has_brand', start_time=start_time)),
        ('ohe', OneHotEncoder())
    ])),
    ('shipping', Pipeline([
        ('select', ItemSelector('shipping', start_time=start_time)),
        ('ohe', OneHotEncoder())
    ])),
    ('item_condition_id', Pipeline([
        ('select', ItemSelector('item_condition_id', start_time=start_time)),
        ('ohe', OneHotEncoder())
    ])),
    ('item_description', Pipeline([
        ('select', ItemSelector('item_description', start_time=start_time)),
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

train_X = vectorizer.fit_transform(train)
print('[{}] Merge vectorized'.format(time() - start_time))
print(train_X.shape)

tfidf_transformer = TfidfTransformer()

train_X = tfidf_transformer.fit_transform(train_X)
print('[{}] TF/IDF completed'.format(time() - start_time))

ridge = Ridge(solver='auto', fit_intercept=True, alpha=0.4, max_iter=200, normalize=False, tol=0.01)
ridge.fit(train_X, train_y)
print('[{}] Train Ridge completed. Iterations: {}'.format(time() - start_time, ridge.n_iter_))
if develop:
    dev, _, _ = preprocess_pandas(dev, start_time, train = False, ss1 = ss1, ss2 = ss2)
    dev_X = vectorizer.transform(dev)
    dev_X = tfidf_transformer.transform(dev_X)
    predsR = ridge.predict(dev_X)
    print("ridge dev RMSLE:", get_rmsle(dev_y, predsR))
else:
    ridge_submission = []
    for test in pd.read_csv(test_file, sep='\t', dtype={'item_condition_id': 'category', 'shipping': 'category'}, engine='c', chunksize=700000):
        test, _, _ = preprocess_pandas(test, start_time, train = False, ss1 = ss1, ss2 = ss2)
        test_X = vectorizer.transform(test)
        test_X = tfidf_transformer.transform(test_X)
        predsR = ridge.predict(test_X)
        submission = test[['test_id']]
        submission['price'] = predsR
        ridge_submission.append(submission)
        del test_X, test
        gc.collect()
del train_X, train_y, train
gc.collect()

############### RNN ###############
print("starting RNN")
import gc
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, Normalizer
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

import sys
import os
import random
import numpy as np
import tensorflow as tf
# os.environ['PYTHONHASHSEED'] = '10000'
# np.random.seed(10001)
# random.seed(10002)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=5, inter_op_parallelism_threads=1)
from keras import backend as K
# tf.set_random_seed(10003)
K.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))
# #Add https://www.kaggle.com/anttip/wordbatch to your kernel Data Sources, 
# #until Kaggle admins fix the wordbatch pip package installation

from nltk.corpus import stopwords
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K

NUM_BRANDS = 4500
NUM_CATEGORIES = 1250


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("missing", "missing", "missing")


def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='No description yet', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'


def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')

# get name and description lengths
def wordCount(text):
    try:
        if text == 'No description yet':
            return 0
        else:
            text = text.lower()
            words = [w for w in text.split(" ")]
            return len(words)
    except: 
        return 0

def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])

def normalize_dataset_text(dataset):
    dataset['item_description'] = dataset['item_description'].apply(lambda x: normalize_text(x))
    dataset['brand_name'] = dataset['brand_name'].apply(lambda x: normalize_text(x))

def delete_unseen(dataset):
    dataset.loc[~dataset['brand_name'].isin(all_brand), 'brand_name'] = 'missing'
    dataset.loc[~dataset['general_cat'].isin(all_general_cat), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(all_subcat_1), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(all_subcat_2), 'subcat_2'] = 'missing'

def text_length_feature(dataset, train = True):
    if train:
        dataset['desc_len'] = dataset['item_description'].apply(lambda x: wordCount(x))
        dataset['name_len'] = dataset['name'].apply(lambda x: wordCount(x))
        dataset[['desc_len', 'name_len']] = desc_normalizer.fit_transform(dataset[['desc_len', 'name_len']])
    else:
        dataset['desc_len'] = dataset['item_description'].apply(lambda x: wordCount(x))
        dataset['name_len'] = dataset['name'].apply(lambda x: wordCount(x))
        dataset[['desc_len', 'name_len']] = desc_normalizer.transform(dataset[['desc_len', 'name_len']])

from time import gmtime, strftime

train = pd.read_table(train_file, engine='c')
train = train[train.price > 0.0].reset_index(drop=True)
if develop:
    train, dev = train_test_split(train, test_size=0.05, random_state = random_seed)
    dev_y = np.log1p(dev["price"])
train_y = np.log1p(train["price"])
print('[{}] Finished to load data'.format(time.time() - start_time))
print('Train shape: ', train.shape)

train['general_cat'], train['subcat_1'], train['subcat_2'] = \
    zip(*train['category_name'].apply(lambda x: split_cat(x)))
train.drop('category_name', axis=1, inplace=True)
print('[{}] Split categories completed.'.format(time.time() - start_time))

handle_missing_inplace(train)
print('[{}] Handle missing completed.'.format(time.time() - start_time))

cutting(train)
print('[{}] Cut completed.'.format(time.time() - start_time))

to_categorical(train)
print('[{}] Convert categorical completed'.format(time.time() - start_time))

desc_normalizer = Normalizer()
name_normalizer = Normalizer()
text_length_feature(train)
print('[{}] Calculate length features'.format(time.time() - start_time))

normalize_dataset_text(train)
print('[{}] Normalization text'.format(time.time() - start_time))

## get all categorical in train and replace missing value
all_brand = set(train["brand_name"].values)
all_general_cat = set(train["general_cat"].values)
all_subcat_1 = set(train["subcat_1"].values)
all_subcat_2 = set(train["subcat_2"].values)

le_brand = LabelEncoder()
le_general_cat = LabelEncoder()
le_subcat_1 = LabelEncoder()
le_subcat_2 = LabelEncoder()

le_brand.fit(train['brand_name'])
train['encoded_brand_name'] = le_brand.transform(train['brand_name'])

le_general_cat.fit(train['general_cat'])
train['encoded_general_cat'] = le_general_cat.transform(train['general_cat'])

le_subcat_1.fit(train['subcat_1'])
train['encoded_subcat_1'] = le_subcat_1.transform(train['subcat_1'])

le_subcat_2.fit(train['subcat_2'])
train['encoded_subcat_2'] = le_subcat_2.transform(train['subcat_2'])

print("Tokenizing item description")
tok_desc = Tokenizer()
tok_desc.fit_on_texts(train["item_description"].values)

print("Tokenizing name")
tok_name = Tokenizer()
tok_name.fit_on_texts(train["name"].values)

print("Transforming text to sequences...")
train['seq_item_description'] = tok_desc.texts_to_sequences(train["item_description"].values)
train['seq_name'] = tok_name.texts_to_sequences(train["name"].values)

## padding max length
MAX_NAME_SEQ = 15 #17
MAX_ITEM_DESC_SEQ = 50 #269

## embedding max length
MAX_DESC_TEXT = len(tok_desc.word_index) + 1
MAX_NAME_TEXT = len(tok_name.word_index) + 1
MAX_BRAND = len(le_brand.classes_)
MAX_GENCAT = len(le_general_cat.classes_)
MAX_SUBCAT_1 = len(le_subcat_1.classes_)
MAX_SUBCAT_2 = len(le_subcat_2.classes_)
MAX_CONDITION = max(train.item_condition_id) + 1

def get_rnn_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        'brand_name': np.array(dataset.encoded_brand_name),
        'item_condition': np.array(dataset.item_condition_id),
        'num_vars': np.array(dataset[["shipping"]]),
        'desc_len': np.array(dataset[["desc_len"]]),
        'name_len': np.array(dataset[["name_len"]]),
        'general_cat': np.array(dataset.encoded_general_cat),
        'subcat_1': np.array(dataset.encoded_subcat_1),
        'subcat_2': np.array(dataset.encoded_subcat_2),
    }
    return X

train_X = get_rnn_data(train)

## RNN Model
np.random.seed(123)

def rnn_model(lr=0.001, decay=0.0):
    # Inputs
    name = Input(shape=[MAX_NAME_SEQ], name="name")
    item_desc = Input(shape=[MAX_ITEM_DESC_SEQ], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    general_cat = Input(shape=[1], name="general_cat")
    subcat_1 = Input(shape=[1], name="subcat_1")
    subcat_2 = Input(shape=[1], name="subcat_2")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[1], name="num_vars")
    desc_len = Input(shape=[1], name="desc_len")
    name_len = Input(shape=[1], name="name_len")

    # Embeddings layers (adjust outputs to help model)
    emb_name = Embedding(MAX_NAME_TEXT, 30)(name)
    emb_item_desc = Dropout(0.05) (Embedding(MAX_DESC_TEXT, 60)(item_desc))
    emb_brand_name = Embedding(MAX_BRAND, 20)(brand_name)
    emb_general_cat = Embedding(MAX_GENCAT, 5)(general_cat)
    emb_subcat_1 = Embedding(MAX_SUBCAT_1, 10)(subcat_1)
    emb_subcat_2 = Embedding(MAX_SUBCAT_2, 15)(subcat_2)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
#     emb_shipping = Embedding(2, 5)(num_vars)
    

    # rnn layers (GRUs are faster than LSTMs and speed is important here)
    rnn_layer1 = GRU(16) (emb_item_desc)
    rnn_layer2 = GRU(8) (emb_name)
    # main layers
    main_l = concatenate([
        Flatten() (emb_brand_name)
        , Flatten() (emb_item_condition)
        , Flatten() (emb_general_cat)
        , Flatten() (emb_subcat_1)
        , Flatten() (emb_subcat_2)
#         , Flatten() (emb_shipping)
        , num_vars
        , rnn_layer1
        , rnn_layer2
        , desc_len
        , name_len
    ])
    # (incressing the nodes or adding layers does not effect the time quite as much as the rnn layers)
    main_l = Dropout(0.05)(Dense(512,kernel_initializer='normal',activation='relu') (main_l))
    main_l = Dropout(0.05)(Dense(96,kernel_initializer='normal',activation='relu') (main_l))

    # the output layer.
    output = Dense(1, activation="linear") (main_l)
    
    model = Model([name, item_desc, brand_name,
                   general_cat, subcat_1, subcat_2,
                   item_condition, num_vars, desc_len, name_len], output)

    optimizer = Adam(lr=lr, decay=decay)
    # (mean squared error loss function works as well as custom functions)  
    model.compile(loss = 'mse', optimizer = optimizer)

    return model

# Set hyper parameters for the model.
BATCH_SIZE = 512 * 4
epochs = 3

# Calculate learning rate decay.
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(train_X['name']) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.007, 0.002
lr_decay = exp_decay(lr_init, lr_fin, steps)

# Create model and fit it with training dataset.
model = rnn_model(lr=lr_init, decay=lr_decay)
checkpointer = ModelCheckpoint(filepath='/tmp/weights.{epoch:02d}.hdf5', verbose=1, save_best_only=False)
model.fit(train_X, train_y, epochs=epochs, batch_size=BATCH_SIZE, verbose=2, callbacks=[checkpointer])


############ Prediction ############
if develop:
    dev['general_cat'], dev['subcat_1'], dev['subcat_2'] = \
        zip(*dev['category_name'].apply(lambda x: split_cat(x)))
    handle_missing_inplace(dev)
    #cutting(dev)
    text_length_feature(dev)
    normalize_dataset_text(dev)
    delete_unseen(dev)
    to_categorical(dev)
    
    ## RNN
    dev['encoded_brand_name'] = le_brand.transform(dev['brand_name'])
    dev['encoded_general_cat'] = le_general_cat.transform(dev['general_cat'])
    dev['encoded_subcat_1'] = le_subcat_1.transform(dev['subcat_1'])
    dev['encoded_subcat_2'] = le_subcat_2.transform(dev['subcat_2'])
    
    dev['seq_item_description'] = tok_desc.texts_to_sequences(dev["item_description"].values)
    dev['seq_name'] = tok_name.texts_to_sequences(dev["name"].values)
    
    dev_X = get_rnn_data(dev)
    preds_rnn = model.predict(dev_X)
    print("RNN dev RMSLE:", rmsle(np.expm1(dev_y), np.expm1(preds_rnn.flatten())))
    
    rnn_epoch2 = rnn_model()
    rnn_epoch2.load_weights("/tmp/weights.02.hdf5")
    rnn_epoch1 = rnn_model()
    rnn_epoch1.load_weights("/tmp/weights.01.hdf5")
    preds_epoch2 = rnn_epoch2.predict(dev_X, batch_size = 10000)
    preds_epoch1 = rnn_epoch1.predict(dev_X, batch_size = 10000)
    
    print("RNN dev RMSLE:", rmsle(np.expm1(dev_y), np.expm1(preds_epoch2.flatten())))
    print("RNN dev RMSLE:", rmsle(np.expm1(dev_y), np.expm1(preds_epoch1.flatten())))
    
    preds_rnn_ensemble = preds_rnn * 0.55 + preds_epoch2 * 0.25 + preds_epoch1 * 0.2
    
    print("RNN dev RMSLE:", rmsle(np.expm1(dev_y), np.expm1(preds_rnn_ensemble.flatten())))
else:
    rnn_submission = []
    for test in pd.read_csv(test_file, sep='\t', chunksize=700000):
        test['general_cat'], test['subcat_1'], test['subcat_2'] = \
            zip(*test['category_name'].apply(lambda x: split_cat(x)))
        test.drop('category_name', axis=1, inplace=True)
        handle_missing_inplace(test)
        #cutting(test)
        text_length_feature(test)
        normalize_dataset_text(test)
        delete_unseen(test)
        to_categorical(test)
        
        ## RNN
        test['encoded_brand_name'] = le_brand.transform(test['brand_name'])
        test['encoded_general_cat'] = le_general_cat.transform(test['general_cat'])
        test['encoded_subcat_1'] = le_subcat_1.transform(test['subcat_1'])
        test['encoded_subcat_2'] = le_subcat_2.transform(test['subcat_2'])
        
        test['seq_item_description'] = tok_desc.texts_to_sequences(test["item_description"].values)
        test['seq_name'] = tok_name.texts_to_sequences(test["name"].values)
        
        test_X = get_rnn_data(test)
        print('[{}] Test data pre-processing'.format(time.time() - start_time))
        preds_rnn = model.predict(test_X, batch_size = 70000)
        print('[{}] RNN prediction'.format(time.time() - start_time))
        temp_model = rnn_model()
        temp_model.load_weights("/tmp/weights.02.hdf5")
        preds_epoch2 = temp_model.predict(test_X, batch_size = 70000)
        temp_model = rnn_model()
        temp_model.load_weights("/tmp/weights.01.hdf5")
        preds_epoch1 = temp_model.predict(test_X, batch_size = 70000)
        preds_rnn_ensemble = preds_rnn * 0.55 + preds_epoch2 * 0.25 + preds_epoch1 * 0.2
        print('[{}] RNN ensemble completed'.format(time.time() - start_time))
        submission = test[['test_id']]
        submission['price'] = preds_rnn_ensemble
        rnn_submission.append(submission)

test_submission = []
for i in range(len(rnn_submission)):
    pred = ridge_submission[i][["test_id"]]
    pred['price'] = np.expm1(ridge_submission[i]['price'] * 0.6 + rnn_submission[i]['price'] * 0.4)
    test_submission.append(pred)
if len(test_submission) > 1:
    test_submission = pd.concat(test_submission)
    test_submission.loc[test_submission['price'] < 0.0, 'price'] = 0.0
    test_submission.to_csv("submission_total.csv", index = False)
else:
    test_submission[0].loc[test_submission[0]['price'] < 0.0, 'price'] = 0.0
    test_submission[0].to_csv("submission_total.csv", index = False)
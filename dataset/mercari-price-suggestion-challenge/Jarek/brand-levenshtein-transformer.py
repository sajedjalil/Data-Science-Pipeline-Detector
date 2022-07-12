# This is some messy code that incorporates brand
# inferring based on Levenshtein as part
# of pipeline, you can diregards most of it
# but for completeness I leave the whole code here
# it will exceed runtime limit in current config
# ufnortunately

import pyximport;
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

pyximport.install()
import time

from wordbatch.models import FM_FTRL
from nltk.corpus import stopwords
from Levenshtein import distance as dameraulevenshtein

from sklearn.feature_extraction.text import TfidfVectorizer

import os

os.environ['JOBLIB_TEMP_FOLDER'] = '.'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['JOBLIB_START_METHOD'] = 'forkserver'
develop = True

import pandas as pd
import os
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import re

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['JOBLIB_START_METHOD'] = 'forkserver'

INPUT_PATH = r'../input'

class SymSpell:
    def __init__(self, max_edit_distance=3, verbose=0):
        self.max_edit_distance = max_edit_distance
        self.verbose = verbose

        self.dictionary = {}
        self.longest_word_length = 0

    def get_deletes_list(self, w):
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
        new_real_word_added = False
        if w in self.dictionary:
            # increment count of word in corpus
            self.dictionary[w] = (self.dictionary[w][0], self.dictionary[w][1] + 1)
        else:
            self.dictionary[w] = ([], 1)
            self.longest_word_length = max(self.longest_word_length, len(w))

        if self.dictionary[w][1] == 1:
            new_real_word_added = True
            deletes = self.get_deletes_list(w)
            for item in deletes:
                if item in self.dictionary:
                    self.dictionary[item][0].append(w)
                else:
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

        return self.dictionary

    def get_suggestions(self, string, silent=False):
        if (len(string) - self.longest_word_length) > self.max_edit_distance:
            if not silent:
                pass
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
            pass
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

    def best_word(self, s, silent=False):
        try:
            return self.get_suggestions(s, silent)[0]
        except:
            return None


class BrandExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.brand_word = r"[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+"

    def fit(self, X, y=None):
        vc = X['brand_name'].value_counts()
        brands = vc[vc > 0].index

        many_w_brands = brands[brands.str.contains(' ')]
        one_w_brands = brands[~brands.str.contains(' ')]

        self.ss2 = SymSpell(max_edit_distance=0)
        self.ss2.create_dictionary_from_arr(many_w_brands, token_pattern=r'.+')

        self.ss1 = SymSpell(max_edit_distance=0)
        self.ss1.create_dictionary_from_arr(one_w_brands, token_pattern=r'.+')
        return self

    def transform(self, X):
        two_words_re = re.compile(r"(?=(\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+))")

        def find_in_str_ss2(row):
            for doc_word in two_words_re.finditer(row):
                suggestion = self.ss2.best_word(doc_word.group(1), silent=True)
                if suggestion is not None:
                    return doc_word.group(1)
            return ''

        def find_in_list_ss1(list_):
            for doc_word in list_:
                suggestion = self.ss1.best_word(doc_word, silent=True)
                if suggestion is not None:
                    return doc_word
            return ''

        def find_in_list_ss2(list_):
            for doc_word in list_:
                suggestion = self.ss2.best_word(doc_word, silent=True)
                if suggestion is not None:
                    return doc_word
            return ''

        X_brand = X['brand_name'].fillna('').copy()


        n_name = X[X_brand == '']['name'].str.findall(
            pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")

        X_brand.loc[X_brand == ''] = [find_in_list_ss2(row) for row in n_name]

        n_desc = X[X_brand == '']['item_description'].str.findall(
            pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
        X_brand.loc[X_brand == ''] = [find_in_list_ss2(row) for row in n_desc]

        n_name = X[X_brand == '']['name'].str.findall(pat=self.brand_word)
        X_brand.loc[X_brand == ''] = [find_in_list_ss1(row) for row in n_name]

        desc_lower = X[X_brand == '']['item_description'].str.findall(pat=self.brand_word)
        X_brand.loc[X_brand == ''] = [find_in_list_ss1(row) for row in desc_lower]

        X_brand.loc[X_brand == ''] = 'noname'


        return X_brand


###########################3

class Preprocess(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X['category_name'] = X['category_name'] \
            .fillna('other/other/other') \
            .astype(str) \
            .str.lower()

        X['name'] = X['name'] \
            .fillna('') \
            .astype(str) \
            .str.lower()

        X['brand_name'] = X['brand_name'] \
            .fillna('') \
            .astype(str) \
            .str.lower()

        X['item_description'] = X['item_description'] \
            .fillna('') \
            .astype(str) \
            .str.lower()

        return X


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class FillNa(BaseEstimator, TransformerMixin):
    def __init__(self, value):
        self.value = value

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X.fillna(self.value)


class LoggingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, message):
        self.message = message

    # noinspection PyUnusedLocal
    def fit(self, X, y=None):
        print("{}: {}, dataset shape: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), self.message,
                                                 X.shape))
        return self

    # noinspection PyMethodMayBeStatic
    def transform(self, X):
        return X


class TopVariance:
    def __init__(self, num_features):
        self.num_features = num_features
        self.vt = VarianceThreshold()

    def fit(self, X, y=None):
        self.vt = self.vt.fit(X)
        if X.shape[1] <= self.num_features:
            threshold = 0.0
        else:
            threshold = np.partition(self.vt.variances_, -self.num_features)[-self.num_features]
        print(threshold)
        self.mask = (self.vt.variances_ > threshold)
        return self

    def transform(self, X):
        return X[:, self.mask]


class LowAlphaNum(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.apply(
            lambda x: ''.join(
                [chr for chr in x.lower() if chr in 'qwertyuiopasdfghjklzxcvbnm1234567890']) if isinstance(x,
                                                                                                           str)
            else '')


class NnzTransformer:
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.mask = np.array(np.clip(X.getnnz(axis=0) - self.threshold, 0, 1), dtype=bool)
        return self

    def transform(self, X):
        return X[:, self.mask]


VARIANCE_THRESHOLD = 2.86738786462e-08


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


pipeline = lambda: Pipeline([
    ('log', LoggingTransformer("Starting transformer/classification pipeline")),
    ('preprocess', Preprocess()),
    ('first_union', FeatureUnion([
        ('name', Pipeline([
            ('selector', ItemSelector('name')),
            ('transformer', FillNa('Unknown')),
            ('vectorizer',
             TfidfVectorizer(use_idf=False, min_df=3, token_pattern=r"(?u)\b\w+\b", analyzer='word',
                             ngram_range=(1, 3), strip_accents='ascii', stop_words=stopwords.words('english'))),
            ('log', LoggingTransformer("End of name")),
            ('vt', VarianceThreshold(VARIANCE_THRESHOLD)),
            ('log2', LoggingTransformer("End of name vt"))
        ])),
        ('name2', Pipeline([
            ('selector', ItemSelector('name')),
            ('transformer', FillNa('null')),
            ('lowalphanum', LowAlphaNum()),
            ('vectorizer',
             TfidfVectorizer(use_idf=False, min_df=3, analyzer='char', ngram_range=(2, 6), strip_accents='ascii')),
            ('log', LoggingTransformer("End of name char")),
            ('vt', VarianceThreshold(VARIANCE_THRESHOLD)),
            ('log2', LoggingTransformer("End of name char vt"))
        ])),
        ('item_description', Pipeline([
            ('selector', ItemSelector('item_description')),
            ('transformer', FillNa('No description yet')),
            ('vectorizer',
             TfidfVectorizer(use_idf=False, min_df=3, token_pattern=r"(?u)\b\w+\b", analyzer='word',
                             ngram_range=(1, 3), strip_accents='ascii', stop_words=stopwords.words('english'))),
            ('log', LoggingTransformer("End of item_description")),
            ('vt', VarianceThreshold(VARIANCE_THRESHOLD)),
            ('log2', LoggingTransformer("End of item_description vt"))
        ])),
        ('item_condition_id', Pipeline([
            ('selector', ItemSelector(['item_condition_id'])),
            ('transformer', FillNa(9999)),
            ('ohe', OneHotEncoder(handle_unknown='ignore')),
            ('log', LoggingTransformer("End of item_condition_id"))
        ])),
        ('item_condition_id_2', Pipeline([
            ('selector', ItemSelector(['item_condition_id'])),
            ('log', LoggingTransformer("End of item_condition_id_2"))
        ])),
        ('category_name', Pipeline([
            ('selector', ItemSelector('category_name')),
            ('transformer', FillNa('Unknown')),
            ('vectorizer',
             CountVectorizer(token_pattern=r"[^/]+", min_df=3, binary=True, analyzer='word',
                             ngram_range=(1, 5), strip_accents='ascii')),
            ('log', LoggingTransformer("End of category_name"))
        ])),
        ('category_name_2', Pipeline([
            ('selector', ItemSelector('category_name')),
            ('transformer', FillNa('Unknown')),
            ('vectorizer',
             CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=3, binary=True, analyzer='word',
                             ngram_range=(1, 2), strip_accents='ascii')),
            ('log', LoggingTransformer("End of category_name 2"))
        ])),
        ('brand_name', Pipeline([
            ('be', BrandExtractor()),
            ('vectorizer', CountVectorizer(token_pattern=r".+", min_df=3, binary=True, analyzer='word',
                                           ngram_range=(1, 1), strip_accents='ascii')),
            ('log', LoggingTransformer("End of brand"))
        ])),
        ('brand_name_2', Pipeline([
            ('selector', ItemSelector('brand_name')),
            ('transformer', FillNa('Unknown')),
            ('vectorizer', CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=3, binary=True, analyzer='word',
                                           ngram_range=(1, 2), strip_accents='ascii')),
            ('log', LoggingTransformer("End of brand 2"))
        ])),
        ('shipping', Pipeline([
            ('selector', ItemSelector(['shipping'])),
            ('log', LoggingTransformer("End of shipping"))
        ]))
    ], n_jobs=1)),
    ('log2', LoggingTransformer("End of first union")),
    ('vt', VarianceThreshold(VARIANCE_THRESHOLD)),
    # ('var', TopVariance(3000000)),
    ('log3', LoggingTransformer("Variance eliminated, end of pipeline")),
]
)


def main():
    start_time = time.time()

    print('[{}] Just started :-)'.format(time.time() - start_time))
    train = pd.read_table('../input/train.tsv', engine='c')
    print('Num zero price {}'.format((train.price < 1.0).sum()))
    train = train.drop(train[(train.price < 1.0)].index)
    print('[{}] Finished to load train data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)

    y = np.log1p(train["price"])
    y_mean = np.mean(y)

    pp = pipeline().fit(train, y)
    print('[{}] Pipeline fitted'.format(time.time() - start_time))
    X = pp.transform(train)
    print('[{}] Pipeline transformed'.format(time.time() - start_time))

    train_X, train_y = X, y
    valid_X, valid_y = None, None

    if develop:
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, shuffle=False, test_size=0.05, random_state=100)

    #### FM FTRL ####
    print('[{}] FM FTRL start training'.format(time.time() - start_time))
    fm_ftrl_model = FM_FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=0.1, D=train_X.shape[1], alpha_fm=0.01, L2_fm=0.0,
                            init_fm=0.01,
                            D_fm=200, e_noise=0.0001, iters=18, inv_link="identity", threads=4)
    fm_ftrl_model.fit(train_X, train_y)
    print('[{}] FM FTRL stop training'.format(time.time() - start_time))
    if develop:
        fm_ftrl_preds = fm_ftrl_model.predict(X=valid_X)
        fm_ftrl_preds = np.clip(fm_ftrl_preds, 0, fm_ftrl_preds.max())
        print("[{}] FM FFTRL dev RMSLE:".format(time.time() - start_time),
              rmsle(np.expm1(valid_y), np.expm1(fm_ftrl_preds)))

    # CREATING SUBMISSION
    def predict():
        for test in pd.read_table('../input/test.tsv', engine='c', chunksize=100000):
            try:
                test_X = pp.transform(test)
                preds = fm_ftrl_model.predict(test_X)
                preds = np.clip(preds, 0, preds.max())

                submission = test[['test_id']]
                submission['price'] = np.expm1(preds)
                yield submission
            except:
                submission_mini_chunks = []
                for i in range(test.shape[0]):
                    test_mini_chunk = test.iloc[i:i + 1]
                    try:
                        test_X_mini_chunk = pp.transform(test_mini_chunk)
                        preds_mini_chunk = fm_ftrl_model.predict(test_X_mini_chunk)
                        preds_mini_chunk = np.clip(preds_mini_chunk, 0, preds_mini_chunk.max())
                    except:
                        preds_mini_chunk = [y_mean]

                    submission_mini_chunk = test_mini_chunk[['test_id']]
                    submission_mini_chunk['price'] = np.expm1(preds_mini_chunk)
                    submission_mini_chunks.append(submission_mini_chunk)

                yield pd.concat(submission_mini_chunks)

    print('[{}] Preparing submission'.format(time.time() - start_time))
    final_submission = pd.concat(predict())
    # noinspection PyUnresolvedReferences
    final_submission.to_csv("submission_fm_ftrl_pipeline_chunks.csv", index=False)

    print('[{}] Submission saved'.format(time.time() - start_time))


if __name__ == '__main__':
    main()

# coding: utf-8

# In[1]:


# coding: utf-8

# In[1]:


import gc
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
import lightgbm as lgb
from bisect import bisect_left, bisect_right
import sys
from operator import itemgetter
from six import string_types
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from scipy.sparse import issparse
#Add https://www.kaggle.com/anttip/wordbatch to your kernel Data Sources, 
#until Kaggle admins fix the wordbatch pip package installation
# sys.path.insert(0, '../input/wordbatch/wordbatch/')
import wordbatch

from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL

# from nltk.corpus import stopwords
import re
from scipy import sparse as ssp
from nltk.corpus import stopwords

import os
from collections import Counter
import multiprocessing
from contextlib import closing
os.environ['OMP_NUM_THREADS'] = '4'
# os.environ['JOBLIB_START_METHOD'] = 'forkserver'

# In[3]:
start_time = time.time()

NUM_BRANDS = 4500
NUM_CATEGORIES = 1200

develop = False
# develop= True

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))

ENGLISH_STOP_WORDS = frozenset([
    'its', 're', 'via', 'inc', 
    'is', 'her', 'if', 'it', 'i',
    'she', 'nor', 'con', 'a', 
    'us', 'me', 'an',
    'eg', 'we', 'how', 'un',
    'who', 'he', 'him', 
    'ie', 'de',
    'the',
    
    'of', 'as', 'our',
])

# Define helpers for text normalization
# stopwords = {x: 1 for x in stopwords.words('english')}
# non_alphanums = re.compile(u'[^A-Za-z0-9]+')

word_regex = re.compile(r"""
    (?:[^\W\d_](?:[^\W\d_]|['\-_])+[^\W\d_]) # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """, re.VERBOSE 
#                         | re.I
                     | re.UNICODE)

#     |
#     (?:\.(?:\s*\.){1,})            # Ellipsis dots.
hang_regex = re.compile(r'([^a-zA-Z0-9])\1{3,}')
rep_regex = re.compile(r"(.)\1{2,}")
def normalize_text(x):
    return u" ".join([x for x in 
        word_regex.findall(hang_regex.sub(r'\1\1\1', 
            rep_regex.sub(r'\1\1\1', 
                x.lower().strip())))
        if len(x) > 1 and x not in ENGLISH_STOP_WORDS])
    
def tokenizer(x):
    return word_regex.findall(hang_regex.sub(r'\1\1\1', rep_regex.sub(r'\1\1\1', x)))
    
# def normalize_text(text):
#     return u" ".join(
#         [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
#          if len(x) > 1 and x not in stopwords])



  
skl_analyzer = HashingVectorizer(
            ngram_range=(1, 2),
            strip_accents='unicode',
            stop_words=ENGLISH_STOP_WORDS,
            tokenizer=tokenizer,
        ).build_analyzer()
def analyzer(x):
    return skl_analyzer(x)     

def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")

class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, fields):
        self.fields = fields

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        return df[self.fields]
        


# In[2]:

class MPCountVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, ngram_weight, seq_l, analyzer=None, max_items=None, min_df=None):
        self.word_counts = {}
        self.word_docs = {}
        self.analyzer = analyzer
        self.seq_l = seq_l
        self.max_items = max_items
        self.min_df = min_df
        self.ngram_weight = ngram_weight
    
    def chunks(self, l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]
                
    def task_multiprocess(self, task, args):
        with closing(multiprocessing.Pool(4, maxtasksperchild=2)) as pool:
            results = pool.map_async(task, args)
            results.wait()
            if results.ready():  
                results = results.get()
        return results
            
    def _dictionary_task(self, batch_docs):
        word_counts = {}
        for doc in batch_docs:
            for tok in self.analyzer(doc):
                if tok in word_counts:
                    word_counts[tok] += 1
                else:
                    word_counts[tok] = 1
        return word_counts
                
    def fit(self, x, y=None):
        print('[{}] Dictionary starting.'.format(time.time() - start_time))
        docs = x.values
#         wc_counts = Counter(self.word_counts)
        wc_counts = Counter()
        wc_dicts = self.task_multiprocess(self._dictionary_task, list(self.chunks(docs, 20000)))
        for wc_d in wc_dicts:
            wc_counts.update(wc_d)
        self.word_counts = dict(wc_counts)
        
        self.make_index(self.max_items, self.min_df)
        print('[{}] Dictionary completed.'.format(time.time() - start_time))
        
        return self
                    
    def make_index(self, max_items=None, min_df=None):
        
        word_counts = self.word_counts
        
        word_counts = sorted(word_counts.items(), key=itemgetter(1))
        if min_df:
            keys, vals = list(zip(*word_counts))
            left_index = bisect_left(vals, min_df)
            word_counts = word_counts[left_index:]
        
        if max_items:
            word_counts = word_counts[-max_features:]
        self.word_index = dict(zip([kv[0] for kv in word_counts], 
                                   range(1, len(word_counts) + 1)))
    
    def _docs_to_sparse(self, batch_docs):
        N_docs = len(batch_docs)
        word_index = self.word_index
        maxl = self.seq_l
        
        j_indices = []
        values = []
        indptr = [0]
        for doc in batch_docs:
            feature_counter = {}
            for tok in self.analyzer(doc):
                try:
                    val = word_index[tok]
                    if val not in feature_counter:
                        feature_counter[val] = self.ngram_weight[0] if len(tok.split()) == 1 else self.ngram_weight[1]
                    else:
                        feature_counter[val] += self.ngram_weight[0] if len(tok.split()) == 1 else self.ngram_weight[1]
                except KeyError:
                    continue
            j_indices += feature_counter.keys()
            values += feature_counter.values()
            indptr.append(len(j_indices))
        
        
        return csr_matrix((values, j_indices, indptr), shape=(len(indptr) - 1, len(word_index)+1), dtype=np.float32)
        
    def _docs_to_seq_and_sparse(self, batch_docs):
        N_docs = len(batch_docs)
        word_index = self.word_index
        maxl = self.seq_l
        ngram_weight = self.ngram_weight
        
        j_indices = []
        values = []
        indptr = [0]
        narr = np.zeros((N_docs, maxl), dtype=np.dtype("i"))
        skip = 0
        for i, doc in enumerate(batch_docs):
            feature_counter = {}
            skip=0
            for j, tok in enumerate(self.analyzer(doc)):
                try:
                    val = word_index[tok]
                    if j < maxl:
                        narr[i, j-skip] = val
                    if val not in feature_counter:
                        feature_counter[val] = ngram_weight[0] if len(tok.split()) == 1 else ngram_weight[1]
                    else:
                        feature_counter[val]  += ngram_weight[0] if len(tok.split()) == 1 else ngram_weight[1]
                except KeyError:
                    skip += 1
                    continue
            j_indices += feature_counter.keys()
            values += feature_counter.values()
            indptr.append(len(j_indices))
            
        return narr, csr_matrix((values, j_indices, indptr), shape=(len(indptr) - 1, len(word_index)+1), dtype=np.float32)
    
    def docs_to_seqs_and_sparse(self, docs):
        batched_data = self.task_multiprocess(self._docs_to_seq_and_sparse, list(self.chunks(docs, 50000)))
        data = list(zip(*batched_data))
        seqs = [item for sublist in data[0] for item in sublist]
        sm = vstack(data[1])
        return seqs, sm
    
    def docs_to_sparse(self, docs, tf_weighting=None):
        batched_seqs = self.task_multiprocess(self._docs_to_sparse, list(self.chunks(docs, 50000)))
        sm = vstack(batched_seqs)
        
        return sm

        
    def transform(self, docs):
        print('[{}] Count vectorization starting.'.format(time.time() - start_time))
        X_seqs, X_sparse = self.docs_to_seqs_and_sparse(docs.values)
        print('[{}] Count vectorization completed.'.format(time.time() - start_time))
        self.X_seqs = X_seqs
        return X_sparse
    
    def get_seqs(self):
        X_seqs = self.X_seqs
        self.X_seqs = None
        return X_seqs

# In[3]:


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, ascategory_names, brands=[]):
        self.ascategory_names = ascategory_names
        self.brands = set(b.lower() for b in brands)
        self.brand_words = set()
        
    def _split_categories(self, df):
        df['gencat_name'], df['subcat1_name'], df['subcat2_name'] = list(zip(*[x.split('/') 
            if isinstance(x, str) else ['missing', 'missing', 'missing'] for x in df['category_name']]))
        df['category_name'] = df['category_name'].apply(lambda x: str(x))
        print('[{}] Split categories completed.'.format(time.time() - start_time))

        return df
    
    def _handle_missing(self, df):
        
        df['name'].fillna('missing', inplace=True)
        df['gencat_name'].fillna('missing', inplace=True)
        df['subcat1_name'].fillna('missing', inplace=True)
        df['subcat2_name'].fillna('missing', inplace=True)
        df['category_name'].fillna('missing', inplace=True)
        df['item_condition_id'].fillna('missing', inplace=True)
        df['shipping'].fillna(0, inplace=True)
        df['item_description'].fillna('missing', inplace=True)
        df['item_description'].replace('No description yet', 'missing', inplace=True)
        df['brand_name'].fillna('missing', inplace=True)
        print('[{}] Handle missing completed.'.format(time.time() - start_time))

        return df
    
    def _create_set(self, x, brand_words):
            x = x.lower().split()
            y = [' '.join(x[i:i+2]) for i in range(len(x))]
            s = set(x+y) & brand_words
            s = list(s)[0] if len(s) == 1 else None
            return s
    def pd_task_multiprocess(self, data, task):
        data_split = np.array_split(data, 16)
        with closing(multiprocessing.Pool(4, maxtasksperchild=2)) as pool:
            results = pool.map_async(task, data_split)
            results.wait(timeout=600)
            if results.ready():  
                results = results.get()
        return pd.concat(results)
    
    def _create_features(self, X):
        regex = re.compile(r' *[\.\?!][\'"\)\]]* +')
        
        X['brand_condition'] = [
                str(p) for p in 
                zip(X['brand_name'].tolist(),
                X['item_condition_id'].tolist())]

        X['brand_condition'].fillna('missing', inplace=True)

        X['brand_shipping'] = [
                str(p) for p in 
                zip(X['brand_name'].tolist(),
                X['shipping'].tolist())]

        X['brand_shipping'].fillna('missing', inplace=True)

        desc_feats = list(zip(*[(len(x), len(max(x.split(), key=len)), len(regex.split(x)),
                                sum([1 for w in x.split() if w[0].isupper()]) / len(x.split()))
                if x != 'missing' else (0, 0, 0, 0) for x in X['item_description']]))
        name_feats = list(zip(*[(len(x), sum(1 for c in x if c.isupper())/len(x), 
                                 sum([1 for w in x.split() if w[0].isupper()]) / len(x.split()))
                if x != 'missing'  else (0, 0, 0) for x in X['name']]))

        X = X.assign(len_desc=desc_feats[0], max_word_desc=desc_feats[1], nsents_desc=desc_feats[2], fstartcaps_desc=desc_feats[3])
        X = X.assign(len_name=name_feats[0], fcaps_name=name_feats[1], fstartcaps_name=name_feats[2])

        X['has_missing'] = X[['gencat_name', 'subcat1_name', 'subcat2_name', 'category_name',
                              'item_condition_id', 'shipping', 'item_description', 'brand_name']
                             ].isin(['missing']).any(axis=1).astype(int)
        
        return X
        
    def fit(self, X, y=None):
        return self.fit_transform(X)
    
    def transform(self, X, y=None):
        check_is_fitted(self, ['_categories', '_brand_words',# 'len_name_bins', 'len_desc_bins'
        ])
        X = self._split_categories(X)
        X = self._handle_missing(X)
        X = self.pd_task_multiprocess(X, self._create_features)
        print('[{}] Additional features created.'.format(time.time() - start_time))

        
        brand_words = self.brand_words
        def create_set(x):
            x = x.lower().split()
            y = [' '.join(x[i:i+2]) for i in range(len(x))]
            s = set(x+y) & brand_words
            s = list(s)[0] if len(s) == 1 else None
            return s
        
        # brand imputation
        brand_words = self._brand_words
        X.loc[X['brand_name'] == 'missing', 'imputed_brand'] = X.loc[X['brand_name'] == 'missing', 'name'].apply(create_set)
        X['imputed_brand'].fillna('missing', inplace=True)
        
        # name and description lengths
        X = self._create_features(X)
        
        categories = self._categories
        X = X.apply(lambda x: pd.Categorical(x, categories[x.name]) if x.name in categories else x, axis=0)
        X = X.apply(lambda x: x.fillna('missing') if x.dtype.name == 'category' else x, axis=0)
        X = X.apply(lambda x: x.cat.codes if x.dtype.name == 'category' else x, axis=0)

        return X
        
    def fit_transform(self, X, y=None):
        
        X = self._split_categories(X)
        X = self._handle_missing(X)
        X = self.pd_task_multiprocess(X, self._create_features)
        print('[{}] Additional features created.'.format(time.time() - start_time))
        
        if len(self.brands) > 0:
            brand_words = self.brands - set(stopwords.words('english')) - set(['missing'])
        else:
            # brand imputation
            brand_words = set(X['brand_name'].str.lower().unique()) - set(stopwords.words('english')) - set(['missing'])
            
        # save brand words for transform
        self._brand_words = brand_words
                
        def create_set(x):
            x = x.lower().split()
            y = [' '.join(x[i:i+2]) for i in range(len(x))]
            s = set(x+y) & brand_words
            s = list(s)[0] if len(s) == 1 else None
            return s

        X.loc[X['brand_name'] == 'missing', 'imputed_brand'] = X.loc[X['brand_name'] == 'missing', 'name'].apply(create_set)
        X['imputed_brand'].fillna('missing', inplace=True)
        
        # convert to categorical
        ascategory_names = self.ascategory_names
        X = X.apply(lambda x: x.astype('category') if x.name in ascategory_names else x, axis=0)
        # make sure category missing is added
        for c in ascategory_names:
            try: cats = X[c].cat.categories.drop('missing')
            except ValueError: cats = X[c].cat.categories
            X[c] = X[c].cat.set_categories(cats.insert(0, 'missing'))
                
        # save categories for transform
        self._categories = {c: X[c].cat.categories for c in X.columns if X[c].dtype.name == 'category'}
        # convert to codes to save memory
        X = X.apply(lambda x: x.cat.codes if x.dtype.name == 'category' else x, axis=0)
        
        return X


class FeatureFilter(BaseEstimator, TransformerMixin):
    def __init__(self, min_feats):
        self.min_feats = min_feats
    
    def fit(self, X, y=None):
        self.mask = np.where(X.getnnz(axis=0) > self.min_feats)[0]
        return self
    
    def transform(self, X, y=None):
        mask = self.mask
        print(X.shape)
        X = X[:, mask]
        print(X.shape)
        return X 


# In[5]:


class LGBMWrapper(BaseEstimator):

    def __init__(self, params, num_boost_round, early_stopping_rounds, verbose_eval, categorical_feature):
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.categorical_feature = categorical_feature
        
    def fit(self, X, y):
        print('[{}] Train LGBM starting'.format(time.time() - start_time))
            
        X = lgb.Dataset(X, label=y, categorical_feature=self.categorical_feature)
        model = lgb.train(self.params, 
                          train_set=X, 
                          num_boost_round=self.num_boost_round,
                          valid_sets=[X],
                          early_stopping_rounds=self.early_stopping_rounds,
                          verbose_eval=self.verbose_eval)
        self.model = model
        self.fitted = True
        print('[{}] Train LGBM completed'.format(time.time() - start_time))
        return self

    def transform(self, X):
        return self.model.predict(X)[:, None]


# In[6]:


class FTRLWrapper(BaseEstimator):

    def __init__(self, alpha, beta, L1, L2, iters, inv_link, threads):
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2
        self.iters = iters
        self.inv_link = inv_link
        self.threads = threads
        self.model = None

    def fit(self, X, y):
        print('[{}] Train FTRL starting'.format(time.time() - start_time))
        model = FTRL(
            alpha=self.alpha, 
            beta=self.beta, 
            L1=self.L1,
            L2=self.L2, 
            D=X.shape[1], 
            iters=self.iters, 
            inv_link=self.inv_link, 
            threads=self.threads)
        model.fit(X, y)
        self.model = model
        print('[{}] Train FTRL completed'.format(time.time() - start_time))
        return self

    def transform(self, X):
        return self.model.predict(X=X)[:, None]


# In[7]:


class FM_FTRLWrapper(BaseEstimator):

    def __init__(self, alpha, beta, L1, L2, alpha_fm, L2_fm, init_fm, D_fm, e_noise, iters, inv_link, threads):
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2
        self.alpha_fm = alpha_fm
        self.L2_fm = L2_fm
        self.init_fm = init_fm
        self.D_fm = D_fm
        self.e_noise = e_noise
        self.iters = iters
        self.inv_link = inv_link
        self.threads = threads
        self.model = None

    def fit(self, X, y):
        print('[{}] Train FM FTRL starting'.format(time.time() - start_time))
        model = FM_FTRL(
            alpha=self.alpha, 
            beta=self.beta,
            L1=self.L1,
            L2=self.L2,
            D=X.shape[1],
            alpha_fm=self.alpha_fm,
            L2_fm=self.L2_fm,
            init_fm=self.init_fm,
            D_fm=self.D_fm,
            e_noise=self.e_noise,
            iters=self.iters,
            inv_link=self.inv_link,
            threads=self.threads)
        model.fit(X, y)
        self.model = model
        print('[{}] Train FM FTRL completed'.format(time.time() - start_time))
        return self

    def transform(self, X):
        return self.model.predict(X=X)[:, None]


# In[8]:




class SparseOneHotEncoder(OneHotEncoder):
    def __init__(self, n_values="auto", categorical_features="all",
            dtype=np.float64, sparse=True, handle_unknown='error'):
        super().__init__(n_values, categorical_features, dtype, sparse, handle_unknown)
    
    def _transform_selected(self, X, transform, selected="all", copy=True):
        """Apply a transform function to portion of selected features
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            Dense array or sparse matrix.
        transform : callable
            A callable transform(X) -> X_transformed
        copy : boolean, optional
            Copy X even if it could be avoided.
        selected: "all" or array of indices or mask
            Specify which features to apply the transform to.
        Returns
        -------
        X : array or sparse matrix, shape=(n_samples, n_features_new)
        """
        X = check_array(X, accept_sparse='csc', copy=copy, dtype=FLOAT_DTYPES)

        if isinstance(selected, string_types) and selected == "all":
            return transform(X)

        if len(selected) == 0:
            return X

        n_features = X.shape[1]
        ind = np.arange(n_features)
        sel = np.zeros(n_features, dtype=bool)
        sel[np.asarray(selected)] = True
        not_sel = np.logical_not(sel)
        n_selected = np.sum(sel)

        if n_selected == 0:
            # No features selected.
            return X
        elif n_selected == n_features:
            # All features selected.
            return transform(X)
        else:
            X_sel = transform(X[:, ind[sel]].toarray())
            X_not_sel = X[:, ind[not_sel]]

            if issparse(X_sel) or issparse(X_not_sel):
                return hstack((X_sel, X_not_sel)).tocsr()
            else:
                return np.hstack((X_sel, X_not_sel))
        
    def transform(self, X, y=None):
        return self._transform_selected(X, self._transform, self.categorical_features, copy=True)
    
    def fit_transform(self, X, y=None):
        return self._transform_selected(X, self._fit_transform, self.categorical_features, copy=True)
            

class FeatureRemover(BaseEstimator, TransformerMixin):
    def __init__(self, column_indices):
        self.column_indices = column_indices
        
    def fit(self, X, y=None):
        n_features = X.shape[1]
        ind = np.arange(n_features)
        sel = np.zeros(n_features, dtype=bool)
        sel[np.asarray(self.column_indices)] = True
        self.mask = np.logical_not(sel)
        return self
    
    def transform(self, X, y=None):
        return X[:, self.mask]
        
# In[9]:

# def main():


# if __name__ == '__main__':
#     main()


# In[2]:


# import tensorflow as tf
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, GRU, Embedding, Flatten, Conv1D, GlobalAveragePooling1D, AveragePooling1D, concatenate
from keras.models import Model
#         from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping#, TensorBoard
from keras import backend as K
from keras import optimizers
from keras import initializers
from keras import regularizers
#         from keras.layers import LeakyReLU

class SequenceCNN(BaseEstimator, TransformerMixin):
    def __init__(self, len_name_vocab, len_desc_vocab, batch_size, epochs, cat_emb_size={}, numeric_names=[]):
        self.len_name_vocab = len_name_vocab
        self.len_desc_vocab = len_desc_vocab
        self.cat_emb_size = cat_emb_size
        self.numeric_names = numeric_names
        self.batch_size = batch_size
        self.epochs = epochs
        
    def _prepare_dataset(self, X):
        dataset = {
            'name': np.vstack(X['name']),
            'item_desc': np.vstack(X['item_description'])
        }
        
        # categorical data
        for key, __ in self.cat_emb_size.items():
            dataset[key] = X[key].values
            
        # numeric data
        def pd_minmaxscaler(df):
            return (df-df.min())/(df.max()-df.min()) - 0.5
        
        for key in self.numeric_names:
            dataset[key] = pd_minmaxscaler(X[key]).values
            
        return dataset
    
    def _prepare_model(self):
        print('defining model')
        max_name_text = self.max_name_text
        max_desc_text = self.max_desc_text
        #Inputs
        name = Input(shape=[20], name="name")
        item_desc = Input(shape=[75], name="item_desc")
        emb_name = Embedding(max_name_text, 15, mask_zero=False)(name)
        emb_item_desc = Embedding(max_desc_text, 30, mask_zero=False)(item_desc)
        
        categorical_inputs = {}
        categorical_embeddings = {}
        numeric_inputs = {}
        for key, val in self.cat_emb_size.items():
            categorical_inputs[key] = Input(shape=[1], name=key)
            categorical_embeddings[key] = Embedding(self.max_cat_codes[key], val)(categorical_inputs[key])
            
        for key in self.numeric_names:
            numeric_inputs[key] = Input(shape=[1], name=key)

        rnn_layer1 = GlobalAveragePooling1D()  (emb_item_desc)
        rnn_layer2 = GlobalAveragePooling1D() (emb_name)

        conv1_desc = Conv1D(filters=4, kernel_size=3,activation='elu'
                           , kernel_regularizer=regularizers.l2(1e-8)) (emb_item_desc)
        conv1_desc = Conv1D(filters=2, kernel_size=3,activation='elu'
                           , kernel_regularizer=regularizers.l2(1e-8)) (conv1_desc)
        conv2_desc = Conv1D(filters=4, kernel_size=3, dilation_rate=2,activation='elu'
                           , kernel_regularizer=regularizers.l2(1e-8)) (emb_item_desc)
        conv1_desc = concatenate([conv1_desc, conv2_desc])
        conv1_desc = Conv1D(filters=2, kernel_size=3,activation='elu'
                           , kernel_regularizer=regularizers.l2(1e-8)) (conv1_desc)
    #     conv3_desc = Conv1D(filters=4, kernel_size=3, dilation_rate=3,activation='relu') (emb_item_desc)
    #     conv1_desc = concatenate([conv1_desc, conv3_desc])
        conv1_desc = GlobalAveragePooling1D() (conv1_desc)

        conv1_name = Conv1D(filters=4, kernel_size=3,activation='elu'
                           , kernel_regularizer=regularizers.l2(1e-8)) (emb_name)
        conv1_name = Conv1D(filters=2, kernel_size=3,activation='elu'
                           , kernel_regularizer=regularizers.l2(1e-8)) (conv1_name)
        conv2_name = Conv1D(filters=4, kernel_size=3, dilation_rate=2,activation='elu'
                           , kernel_regularizer=regularizers.l2(1e-8)) (emb_name)
        conv1_name = concatenate([conv1_name, conv2_name])
        conv1_name = Conv1D(filters=2, kernel_size=3,activation='elu'
                           , kernel_regularizer=regularizers.l2(1e-8)) (conv1_name)
    #     conv3_name = Conv1D(filters=4, kernel_size=3, dilation_rate=3,activation='relu') (emb_name)
    #     conv1_name = concatenate([conv1_name, conv3_name])
        conv1_name = GlobalAveragePooling1D() (conv1_name)

        dnn_brand = Dense(4, kernel_initializer='glorot_uniform',activation='relu'
                         ) (categorical_embeddings['brand_name'])

        #main layer
        main_l = concatenate(
             [
                 Flatten()(categorical_embeddings['gencat_name']),
                 Flatten()(categorical_embeddings['subcat1_name']),
                 Flatten()(categorical_embeddings['subcat2_name']),
                 Flatten()(categorical_embeddings['item_condition_id']),
                 Flatten()(dnn_brand),
                 GlobalAveragePooling1D()(categorical_embeddings['brand_name']),
                 GlobalAveragePooling1D()(categorical_embeddings['imputed_brand']),
             ] + list(numeric_inputs.values()) + [
                rnn_layer1,
                rnn_layer2,
                conv1_desc,
                conv1_name
            ]
        )

        main_l = Dense(256,kernel_initializer='glorot_uniform',activation='relu', kernel_regularizer=regularizers.l2(1e-8)
                      ) (main_l)
        main_l = Dropout(0.2)(main_l)
        main_l = Dense(256,kernel_initializer='glorot_uniform',activation='relu', kernel_regularizer=regularizers.l2(1e-8)
                      ) (main_l)
        main_l = Dropout(0.2)(main_l)

        output = Dense(1, activation='linear') (main_l)
        
        model = Model(
            list(categorical_inputs.values()) + list(numeric_inputs.values()) + [
                name,
                item_desc
            ], output)

        optimizer = optimizers.Adam()
        model.compile(loss="mse", 
                      optimizer=optimizer)
        print('[{}] Finished DEFINING MODEL...'.format(time.time() - start_time))

        return model
    
    def _run_model(self, model, dataset, y, steps):
        
        #fin_lr=init_lr * (1/(1+decay))**(steps-1)
        exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
        
        print('Fitting model')
        #FITTING THE MODEL
        # lr_init, lr_fin = 0.013, 0.009
        # lr_init, lr_fin = 0.005, 0.0005
        lr_init, lr_fin = 0.007, 0.003
        lr_decay = exp_decay(lr_init, lr_fin, steps)

        K.set_value(model.optimizer.lr, lr_init)
        K.set_value(model.optimizer.decay, lr_decay)

        history = model.fit(dataset, y
                            , epochs=self.epochs
                            , batch_size=self.batch_size
                            #, validation_split=0.01
                            , verbose=0
                            )

        print('[{}] Finished FITTING MODEL...'.format(time.time() - start_time))
        
    def _preprocess(self, X):
        len_name_vocab = self.len_name_vocab
        len_desc_vocab = self.len_desc_vocab
        cat_emb_size = self.cat_emb_size
        
        
        max_name_text = len_name_vocab + 1
        max_desc_text = len_desc_vocab + 1
        
        max_cat_codes = {}
        for key, __ in cat_emb_size.items():
            max_cat_codes[key] = X[key].max() + 1
                
        self.max_name_text = max_name_text
        self.max_desc_text = max_desc_text
        self.max_cat_codes = max_cat_codes

    def fit(self, X, y):

        batch_size = self.batch_size
        epochs = self.epochs
        steps = len(X) // batch_size * epochs
        
        self._preprocess(X)
        X = self._prepare_dataset(X)
        model = self._prepare_model()
        self._run_model(model, X, y, steps)

        self.model = model
        
        return self
    
    def transform(self, X, y=None):
        return self.model.predict(self._prepare_dataset(X))


# In[3]:



# import os
# os.system("( awk -F '\t' '{ print $5 }' ../input/train.tsv ; awk -F '\t' '{ print $5 }' ../input/test.tsv ) | sort | uniq | sed '1d' > brands")
# with open('brands', 'rb') as f: brands = f.read().decode().splitlines()

preprocessor = DataPreprocessor(ascategory_names=[
            'brand_condition',
            'brand_shipping',
            'item_condition_id',
            'gencat_name',
            'subcat1_name',
            'subcat2_name',
            'brand_name',
            'imputed_brand',
    #             'len_name',
    #             'len_desc',
    #             'shipping',

    ],
                                    #   brands=brands
                                     )
    
vectorizer = FeatureUnion([ 
        ('onehot_features', ItemSelector([
            'brand_condition',
            'brand_shipping',
            'item_condition_id',
            'gencat_name',
            'subcat1_name',
            'subcat2_name',
            'brand_name',
            'imputed_brand',
            'shipping',
            'has_missing'
        ])),
        ('numeric_features', Pipeline([
            ('selector', ItemSelector(
                fields = [
                    'len_name', 
                    'len_desc',
                    'max_word_desc',
                    'nsents_desc', 
                    'fstartcaps_desc',
                    'fcaps_name', 
                    'fstartcaps_name'
                    ]
                )
            ),
            ('minmaxscaler', MinMaxScaler()),
        ])),
        ('name_cv', Pipeline([
            ('selector', ItemSelector(fields = 'name')), 
            ('cv', MPCountVectorizer(ngram_weight=[1.5, 1.0], seq_l=20, analyzer=analyzer, min_df=4))
        ])), 
        ('item_desc_cv', Pipeline([
            ('selector', ItemSelector(fields = 'item_description')), 
            ('cv', MPCountVectorizer(ngram_weight=[1.0, 1.0], seq_l=75, analyzer=analyzer, min_df=9)), 
            ('tfidf', TfidfTransformer(sublinear_tf = True))
        ])),
        ], n_jobs=1)

model = FeatureUnion([
    ('nonlinear_predictors', Pipeline([
        ('interaction_remover', FeatureRemover(np.array([0, 1, 9]))),
        ('feature_filter', FeatureFilter(100)),
        ('lgbm', LGBMWrapper(
            params = {
                'learning_rate': 0.6,
                'application': 'regression',
                'max_depth': 3,
                'num_leaves': 60,
                'verbosity': -1,
                'metric': 'RMSE',
                'data_random_seed': 1,
                'bagging_fraction': 0.5,
                'nthread': 4,
                'min_data_in_leaf': 200,
                'max_bin': 30,
            },
            num_boost_round = 5000,
            early_stopping_rounds = 250,
            verbose_eval = 1000,
            categorical_feature = list(range(6)),
        ))
    ])),
    ('linear_predictors', Pipeline([
        ('ohe', SparseOneHotEncoder(
            categorical_features=np.arange(8),
            )),
        ('feature_filter', FeatureFilter(2)),
        ('linear_regressors', FeatureUnion([
            # ('ftrl', FTRLWrapper(
            #     alpha = 0.01,
            #     beta = 0.1,
            #     L1 = 0.00001,
            #     L2 = 1.0,
            #     iters = 50,
            #     inv_link = "identity",
            #     threads = 1)), 
            ('fm_ftrl', FM_FTRLWrapper(
                alpha = 0.015123039358983292,
                beta = 0.0036961595959144268,
                L1 = 0.0001071848083393167,
                L2 = 0.15063815187520505,
                alpha_fm = 0.0094245031253530013,
                L2_fm = 0.0,
                init_fm = 0.01,
                D_fm = 157,
                e_noise = 0.00011964508569471388,
                iters = 18,
                inv_link = "identity",
                threads = 4)),
        ])),
    ])), 

])

# del brands


# In[9]:



train = pd.read_table('../input/train.tsv', engine='c')

train = train.drop(train[(train.price < 1.0)].index)

train_X = train

train_y = np.log1p(train_X['price'])


# In[5]:


train_processed = preprocessor.fit_transform(train_X)


# In[6]:


train_vect = vectorizer.fit_transform(train_processed)


# In[7]:



del train, train_X, train_processed['name'], train_processed['item_description']
model.fit(train_vect, train_y)
del train_vect


# In[10]:


train_processed['name'] = vectorizer.transformer_list[2][1].named_steps['cv'].get_seqs()
train_processed['item_description'] = vectorizer.transformer_list[3][1].named_steps['cv'].get_seqs()
len_name_vocab = len(vectorizer.transformer_list[2][1].named_steps['cv'].word_index)
len_desc_vocab = len(vectorizer.transformer_list[3][1].named_steps['cv'].word_index)

cnn = SequenceCNN(len_name_vocab, 
                  len_desc_vocab, 
                  cat_emb_size={
                      'brand_name': 10, 
                      'gencat_name': 5, 
                      'subcat1_name': 5, 
                      'subcat2_name': 5, 
                      'item_condition_id': 3, 
                      'imputed_brand': 10,
                  }, 
                  numeric_names=['len_name', 
                    'len_desc',
                    'max_word_desc',
                    'nsents_desc', 
                    'fstartcaps_desc',
                    'fcaps_name', 
                    'fstartcaps_name',
                    'shipping'],
                  epochs=2,
                  batch_size=512*3
)

cnn.fit(train_processed, train_y)
# del cnn
# gc.collect()
# K.clear_session()


del train_processed, train_y

def load_test():
    for df in pd.read_table('../input/test.tsv', engine='c', chunksize=700000, index_col=0):
        yield df
        
submissions = []
for test in load_test():
    submission = pd.DataFrame(test.index)
    test_processed = preprocessor.transform(test)
    del test
    test_vect = vectorizer.transform(test_processed)
    del test_processed['name'], test_processed['item_description']
    test_preds = model.transform(test_vect)
    del test_vect

    test_processed['name'] = vectorizer.transformer_list[2][1].named_steps['cv'].get_seqs()
    test_processed['item_description'] = vectorizer.transformer_list[3][1].named_steps['cv'].get_seqs()
    test_preds_cnn = cnn.transform(test_processed)
    mask = test_processed['has_missing'].values.astype(bool)
    del test_processed

    test_preds = test_preds[:, 0] * 0.25 + test_preds[:, 1] * 0.35 + test_preds_cnn[:, 0] * 0.4
    
    submission['price'] = np.expm1(test_preds)
    del test_preds
    submissions.append(submission)
    
submissions = pd.concat(submissions)
submissions.loc[submissions['price'] < 0.0, 'price'] = 0.0
submissions.to_csv("submission_wordbatch_ftrl_fm_lgb.csv", index=False)


# In[ ]:
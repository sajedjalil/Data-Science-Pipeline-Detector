import json

import scipy as sp
import pandas as pd
import numpy as np

from functools import partial
from math import sqrt

from sklearn.metrics import cohen_kappa_score, mean_squared_error,log_loss
from sklearn.metrics import confusion_matrix as sk_cmatrix
from sklearn.model_selection import StratifiedKFold,train_test_split,KFold,GroupKFold,StratifiedShuffleSplit
import cv2
from tqdm import tqdm, tqdm_notebook
from sklearn.linear_model import RidgeClassifier,LogisticRegression,SGDClassifier,SGDRegressor,LinearRegression,Ridge,PassiveAggressiveClassifier,PassiveAggressiveRegressor,BayesianRidge
from wordbatch.models import FTRL,FM_FTRL
import wordbatch
from wordbatch.extractors import WordBag
from sklearn.svm import LinearSVR,LinearSVC
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications import InceptionResNetV2, InceptionV3
from keras.applications.densenet import preprocess_input, DenseNet121
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.initializers import *
import keras.backend as K
import tensorflow as tf
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from catboost import Pool, CatBoostClassifier,CatBoostRegressor
import gc
import re
import nltk
from nltk.corpus import stopwords
import string
import jieba
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, vstack
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from scipy.sparse import hstack
import featuretools as ft
import glob
from joblib import Parallel, delayed
from PIL import Image
import pprint
import time
from gensim.models import Word2Vec
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, Imputer
import gensim
from keras.utils import to_categorical
from itertools import product
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(369)
rn.seed(369)
tf.set_random_seed(1234)
img_size = 256
batch_size = 16
embed_size=128
print(os.listdir("../input"))

stops = {x: 1 for x in stopwords.words('english')}
start_time=time.time()


class MeanEncoder:
    def __init__(self, categorical_features, n_splits=5, target_type='classification', prior_weight_func=None):
        """
        :param categorical_features: list of str, the name of the categorical columns to encode
        :param n_splits: the number of splits used in mean encoding
        :param target_type: str, 'regression' or 'classification'
        :param prior_weight_func:
        a function that takes in the number of observations, and outputs prior weight
        when a dict is passed, the default exponential decay function will be used:
        k: the number of observations needed for the posterior to be weighted equally as the prior
        f: larger f --> smaller slope
        '''
        >>>example:
        mean_encoder = MeanEncoder(
                        categorical_features=['regionidcity',
                          'regionidneighborhood', 'regionidzip'],
                target_type='regression'
                )

        X = mean_encoder.fit_transform(X, pd.Series(y))
        X_test = mean_encoder.transform(X_test)


        """

        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}

        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None

        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))

    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()

        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(int)  # classification
        else:
            nf_name = '{}_pred'.format(variable)
            X_train['pred_temp'] = y_train  # regression
        prior = X_train['pred_temp'].mean()

        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({'mean': 'mean', 'beta': 'size'})
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y['mean']
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)

        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values

        return nf_train, nf_test, prior, col_avg_y

    def fit_transform(self, X, y):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)

        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return X_new

    def transform(self, X):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()

        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits

        return X_new

def get_embedding_matrix(word_index,embed_size=embed_size, Emed_path="w2v_128.txt"):
    embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(
        Emed_path, binary=False)
    nb_words = len(word_index)+1
    embedding_matrix = np.zeros((nb_words, embed_size))
    count = 0
    for word, i in tqdm(word_index.items()):
        if i >= nb_words:
            continue
        try:
            embedding_vector = embeddings_index[word]
        except:
            embedding_vector = np.zeros(embed_size)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            count += 1
    return embedding_matrix
    

class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
class AdamW(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4,  # decoupled weight decay (1/4)
                 epsilon=1e-8, decay=0., **kwargs):
        super(AdamW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.wd = K.variable(weight_decay, name='weight_decay') # decoupled weight decay (2/4)
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        wd = self.wd # decoupled weight decay (3/4)

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - lr * wd * p # decoupled weight decay (4/4)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'weight_decay': float(K.get_value(self.wd)),
                  'epsilon': self.epsilon}
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(tf.keras.backend.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = tf.keras.backend.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
    

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

def gelu(x):
    cdf = 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0)))
    return x * cdf
    
def lan_type(desc):
    desc = str(desc)
    if desc=='nan':
        return 0
    zh_model = re.compile(u'[\u4e00-\u9fa5]')    
    en_model = re.compile(u'[a-zA-Z]')  
    zh_match = zh_model.search(desc)
    en_match = en_model.search(desc)
    if zh_match and en_match:
        return 3  
    elif zh_match:
        return 3  
    elif en_match:
        return 2  
    else:
        return 1  

def malai_type(desc):
    desc = str(desc)
    malai = [' la x ' , ' nk ',' nie ', ' umur ', ' di ', 'teruk', ' satu ',' dh ', ' ni ',' tp ', ' yg ', 'mmg', 'msj', ' utk ' ,'neh' ]
    for tag in malai:
        if desc.find(tag) > -1:
            return 1
    
    return  0

def normalize_text(text):
    text = text.lower().strip()
    for s in string.punctuation:
        text = text.replace(s, ' ')
    text = text.strip().split(' ')
    return u' '.join(x for x in text if len(x) > 1 and x not in stops)
    
def resize_to_square(im):
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return new_im

def load_image(path, pet_id):
    image = cv2.imread(f'{path}{pet_id}-1.jpg')
    new_image = resize_to_square(image)
    new_image = preprocess_input(new_image)
    return new_image

def img_model():
    K.clear_session()
    inp = Input((img_size, img_size, 3))
    x = DenseNet121(
            include_top=False, 
            weights="../input/keras-pretrain-model-weights/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5", 
            input_shape=(img_size, img_size, 3))(inp)
    x = GlobalAveragePooling2D()(x)
    x = Lambda(lambda x: K.expand_dims(x, axis = -1))(x)
    x = AveragePooling1D(4)(x)
    out = Lambda(lambda x: x[:, :, 0])(x)

    model = Model(inp, out)
    return model
def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p,weights='quadratic')
        return -ll

    def fit(self, X, y,initial_coef=[0.5, 1.5, 2.5, 3.5]):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
#         initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef,len_0):
        X_p = np.copy(X)
        temp = sorted(list(X_p))
        threshold=temp[int(0.95*len_0)-1]
        for i, pred in enumerate(X_p):
            if pred < threshold:
                X_p[i] = 0
            elif pred >= threshold and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']

def get_class_bounds(y, y_pred, N=5, class0_fraction=-1):
    """
    Find boundary values for y_pred to match the known y class percentiles.
    Returns N-1 boundaries in y_pred values that separate y_pred
    into N classes (0, 1, 2, ..., N-1) with same percentiles as y has.
    Can adjust the fraction in Class 0 by the given factor (>=0), if desired. 
    """
    ysort = np.sort(y)
    predsort = np.sort(y_pred)
    bounds = []
    for ibound in range(N-1):
        iy = len(ysort[ysort <= ibound])
        # adjust the number of class 0 predictions?
        if (ibound == 0) and (class0_fraction >= 0.0) :
            iy = int(class0_fraction * iy)
        bounds.append(predsort[iy])
    return bounds

def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):
    kf = StratifiedKFold(n_splits=5, random_state=1017, shuffle=True)
#     kf = GroupKFold(n_splits=5)
#     fold_splits = kf.split(train, target,group)
#     kf = StratifiedShuffleSplit(n_splits=10, test_size=0.45,  random_state=1017)
    fold_splits = kf.split(train, target)
    folds=5
    cv_scores = []
    qwk_scores = []
    pred_full_test = 0
    log_list=[]
    pred_train = np.zeros((train.shape[0], folds))
    all_coefficients = np.zeros((folds, 4))
    feature_importance_df = pd.DataFrame()
    i = 1
    for dev_index, val_index in fold_splits:
        print( label + ' | FOLD ' + str(i) + '/'+str(folds))
        if isinstance(train, pd.DataFrame):
            dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
            dev_y, val_y = target[dev_index], target[val_index]
        else:
            dev_X, val_X = train[dev_index], train[val_index]
            dev_y, val_y = target[dev_index], target[val_index]
        params2 = params.copy()
        pred_val_y, pred_test_y, importances, coefficients, qwk,log = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        all_coefficients[i-1, :] = coefficients
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
            qwk_scores.append(qwk)
            log_list.append(log)
            print(label + ' cv score {}: RMSE {} QWK {}'.format(i, cv_score, qwk))
            print("##"*40)
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = train.columns.values
        fold_importance_df['importance'] = importances
        fold_importance_df['fold'] = i
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)        
        i += 1
#     print('{} cv RMSE scores : {}'.format(label, cv_scores))
    print('{} cv mean RMSE score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std RMSE score : {}'.format(label, np.std(cv_scores)))
#     print('{} cv QWK scores : {}'.format(label, qwk_scores))
    print('{} cv mean QWK score : {}'.format(label, np.mean(qwk_scores)))
    print('{} cv std QWK score : {}'.format(label, np.std(qwk_scores)))
    print('{} cv mean log_loss score : {}'.format(label, np.mean(log_list)))
    
    pred_full_test = pred_full_test / float(folds)
    results = {'label': label,
               'train': pred_train, 'test': pred_full_test,
                'cv': cv_scores, 'qwk': qwk_scores,
               'importance': feature_importance_df,
               'coefficients': all_coefficients}
    return results

def get_feat1():
    train_data = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")
    test_data = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")
    
    def deal_breed(df):
        if df['Breed1']==df['Breed2']:
            df['Breed2']=0
        if df['Breed1']!=307 & df['Breed2']==307:
            temp=df["Breed1"]
            df['Breed1']=df['Breed2']
            df['Breed2']=temp
        return df
    
    train_data=train_data.apply(lambda x:deal_breed(x),1)
    test_data=test_data.apply(lambda x:deal_breed(x),1)
    
    def get_purebreed_feat(df):
        if df['Breed2']==0 and df['Breed1']!=307:
            return 1
        return 0
    train_data['purebreed']=train_data.apply(lambda x:get_purebreed_feat(x),1)
    test_data['purebreed']=test_data.apply(lambda x:get_purebreed_feat(x),1)
    
    train_data['is_group']=train_data.Gender.apply(lambda x:1 if x==3 else 0,1)
    test_data['is_group']=test_data.Gender.apply(lambda x:1 if x==3 else 0,1)
    
    def get_good_cnt(df):
        cnt=0
        for i in ['Vaccinated',"Dewormed","Sterilized","Health"]:
            if df[i]==1:
                cnt+=1
        return cnt
    
    train_data['good_cnt']=train_data.apply(lambda x:get_good_cnt(x),1)
    test_data['good_cnt']=test_data.apply(lambda x:get_good_cnt(x),1)
    
    
    
    train_data['lan_type'] = train_data.Description.map(lambda x:lan_type(x))
    train_data['malai_type'] = train_data.Description.map(lambda x:malai_type(x))
    
    test_data['lan_type'] = test_data.Description.map(lambda x:lan_type(x))
    test_data['malai_type'] = test_data.Description.map(lambda x:malai_type(x))
    
    def name_deal(df):
        if "No Name" in df:
            return np.nan
        if df =="nan":
            return np.nan
        else:
            return df
    train_data['Name'] = train_data['Name'].apply(lambda x:name_deal(str(x)),1)
    test_data['Name'] = test_data['Name'].apply(lambda x:name_deal(str(x)),1)
    
    train_data['Name'],indexer=pd.factorize(train_data['Name'])
    test_data['Name'] = indexer.get_indexer(test_data['Name'])
    
    rescuer_df=train_data.groupby("RescuerID",as_index=False).count()[["RescuerID","PetID"]]
    rescuer_df.columns=["RescuerID","rescuer_cnt"]
    train_data=pd.merge(train_data,rescuer_df,on="RescuerID",how="left")
    # train_data.drop("RescuerID",1,inplace=True)
    
    rescuer_df=test_data.groupby("RescuerID",as_index=False).count()[["RescuerID","PetID"]]
    rescuer_df.columns=["RescuerID","rescuer_cnt"]
    test_data=pd.merge(test_data,rescuer_df,on="RescuerID",how="left")
    # test_data.drop("RescuerID",1,inplace=True)
    del rescuer_df
    gc.collect()
    
    train_data['rescuer_rank'] = train_data['RescuerID'].map(train_data['RescuerID'].value_counts().rank()/len(train_data['RescuerID'].unique()))
    test_data['rescuer_rank'] = test_data['RescuerID'].map(test_data['RescuerID'].value_counts().rank()/len(test_data['RescuerID'].unique()))
    
    def get_res_feat(df):
        temp=pd.DataFrame(index=range(1))
        temp['RescuerID']=df['RescuerID'].values[0]
        temp['res_type_cnt']=len(df['Type'].unique())
        temp['res_breed_cnt']=len(df['Breed1'].unique())
        temp['res_breed_mode']=df['Breed1'].mode()
        temp['res_fee_mean']=df['Fee'].mean()
        temp['res_Quantity_sum']=df['Quantity'].sum()
        temp['res_MaturitySize_mean']=df['MaturitySize'].mean()
        temp['res_Description_unique']=len(df['Description'].unique())
        return temp
    train_res_feat=train_data.groupby("RescuerID",as_index=False).apply(lambda x:get_res_feat(x))
    test_res_feat=test_data.groupby("RescuerID",as_index=False).apply(lambda x:get_res_feat(x))
    train_res_feat.index=range(len(train_res_feat))
    test_res_feat.index=range(len(test_res_feat))
    train_data = pd.merge(train_data,train_res_feat,on="RescuerID",how="left")
    test_data = pd.merge(test_data,test_res_feat,on="RescuerID",how="left")
    train_data.drop("RescuerID",1,inplace=True)
    test_data.drop("RescuerID",1,inplace=True)
    train_data['fee-mean_fee']=train_data['Fee']-train_data['res_fee_mean']
    test_data['fee-mean_fee']=test_data['Fee']-test_data['res_fee_mean']
    del train_res_feat,test_res_feat
    gc.collect()
    
    train_data['Description'] = train_data['Description'].fillna("null")
    test_data['Description'] = test_data['Description'].fillna("null")
    
    train_data["Description"] = train_data["Description"].str.lower()
    test_data["Description"] = test_data["Description"].str.lower()
    
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
     '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
     '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
     '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
     '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
    def clean_text(x):

        x = str(x)
        for punct in puncts:
            x = x.replace(punct, f' {punct} ')
        return x


    train_data["Description"] = train_data["Description"].apply(lambda x: clean_text(x))
    test_data["Description"] = test_data["Description"].apply(lambda x: clean_text(x))
    
    

    eng_stopwords = set(stopwords.words("english"))
    
    # ## Number of words in the text ##
    train_data["num_words"] = train_data["Description"].apply(lambda x: len(str(x).split()))
    test_data["num_words"] = test_data["Description"].apply(lambda x: len(str(x).split()))
    
    # ## Number of unique words in the text ##
    train_data["num_unique_words"] = train_data["Description"].apply(lambda x: len(set(str(x).split())))
    test_data["num_unique_words"] = test_data["Description"].apply(lambda x: len(set(str(x).split())))
    
    # ## Number of characters in the text ##
    train_data["num_chars"] = train_data["Description"].apply(lambda x: len(str(x)))
    test_data["num_chars"] = test_data["Description"].apply(lambda x: len(str(x)))
    
    # ## Number of stopwords in the text ##
    train_data["num_stopwords"] = train_data["Description"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
    test_data["num_stopwords"] = test_data["Description"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
    
    # ## Number of punctuations in the text ##
    train_data["num_punctuations"] =train_data['Description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
    test_data["num_punctuations"] =test_data['Description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
    
    # ## Number of title case words in the text ##
    # train_data["num_words_upper"] = train_data["Description"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    # test_data["num_words_upper"] = test_data["Description"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    
    # ## Number of title case words in the text ##
    # train_data["num_words_title"] = train_data["Description"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    # test_data["num_words_title"] = test_data["Description"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    
    # ## Average length of the words in the text ##
    train_data["mean_word_len"] = train_data["Description"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    test_data["mean_word_len"] = test_data["Description"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    
    # # train_data['num_vs_len']=train_data['num_punctuations']/train_data['num_chars']
    # # test_data['num_vs_len']=test_data['num_punctuations']/test_data['num_chars']
    
    # # train_data['up_vs_len'] = train_data['num_words_upper'] / train_data['num_words']
    # # test_data['up_vs_len'] = test_data['num_words_upper'] / test_data['num_words']
    
    # # train_data['senten_cnt']=train_data["Description"].apply(lambda x:len(str(x).split(".")),1)
    # # test_data['senten_cnt']=test_data["Description"].apply(lambda x:len(str(x).split(".")),1)
    
    
    def deal_desc(df):
        if df['lan_type']==1:
            return "null"
        if df['lan_type']==3:
            text=jieba.cut(df['Description'])
            text=" ".join(text)
            text=text.replace("   "," ")
            return text
        else:
            return df['Description']
    train_data['Description']=train_data.apply(lambda x:deal_desc(x),1)
    test_data['Description']=test_data.apply(lambda x:deal_desc(x),1)
    
    doc_sent_mag = []
    doc_sent_score = []
    nf_count = 0
    for pet in train_data['PetID'].values:
        try:
            with open('../input/petfinder-adoption-prediction/train_sentiment/' + pet + '.json', 'r') as f:
                sentiment = json.load(f)
            doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
            doc_sent_score.append(sentiment['documentSentiment']['score'])
        except FileNotFoundError:
            nf_count += 1
            doc_sent_mag.append(-1)
            doc_sent_score.append(-1)
    
    train_data['doc_sent_mag'] = doc_sent_mag
    train_data['doc_sent_score'] = doc_sent_score
    
    doc_sent_mag = []
    doc_sent_score = []
    nf_count = 0
    for pet in test_data['PetID'].values:
        try:
            with open('../input/petfinder-adoption-prediction/test_sentiment/' + pet + '.json', 'r') as f:
                sentiment = json.load(f)
            doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
            doc_sent_score.append(sentiment['documentSentiment']['score'])
        except FileNotFoundError:
            nf_count += 1
            doc_sent_mag.append(-1)
            doc_sent_score.append(-1)
    
    test_data['doc_sent_mag'] = doc_sent_mag
    test_data['doc_sent_score'] = doc_sent_score
    del doc_sent_mag,doc_sent_score
    gc.collect()
    
    vertex_xs = []
    vertex_ys = []
    bounding_confidences = []
    bounding_importance_fracs = []
    dominant_blues = []
    dominant_greens = []
    dominant_reds = []
    dominant_pixel_fracs = []
    dominant_scores = []
    label_descriptions = []
    label_scores = []
    nf_count = 0
    nl_count = 0
    for pet in train_data['PetID'].values:
        try:
            with open('../input/petfinder-adoption-prediction/train_metadata/' + pet + '-1.json', 'r') as f:
                data = json.load(f)
            vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
            vertex_xs.append(vertex_x)
            vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
            vertex_ys.append(vertex_y)
            bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
            bounding_confidences.append(bounding_confidence)
            bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
            bounding_importance_fracs.append(bounding_importance_frac)
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('blue',-1)
            dominant_blues.append(dominant_blue)
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('green',-1)
            dominant_greens.append(dominant_green)
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('red',-1)
            dominant_reds.append(dominant_red)
            dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
            dominant_pixel_fracs.append(dominant_pixel_frac)
            dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
            dominant_scores.append(dominant_score)
            if data.get('labelAnnotations'):
                label_description = data['labelAnnotations'][0]['description']
                label_descriptions.append(label_description)
                label_score = data['labelAnnotations'][0]['score']
                label_scores.append(label_score)
            else:
                nl_count += 1
                label_descriptions.append('nothing')
                label_scores.append(-1)
        except FileNotFoundError:
            nf_count += 1
            vertex_xs.append(-1)
            vertex_ys.append(-1)
            bounding_confidences.append(-1)
            bounding_importance_fracs.append(-1)
            dominant_blues.append(-1)
            dominant_greens.append(-1)
            dominant_reds.append(-1)
            dominant_pixel_fracs.append(-1)
            dominant_scores.append(-1)
            label_descriptions.append('nothing')
            label_scores.append(-1)
    
    print(nf_count)
    print(nl_count)
    train_data[ 'vertex_x'] = vertex_xs
    train_data['vertex_y'] = vertex_ys
    train_data['bounding_confidence'] = bounding_confidences
    train_data['bounding_importance'] = bounding_importance_fracs
    train_data['dominant_blue'] = dominant_blues
    train_data['dominant_green'] = dominant_greens
    train_data['dominant_red'] = dominant_reds
    train_data['dominant_pixel_frac'] = dominant_pixel_fracs
    train_data['dominant_score'] = dominant_scores
    train_data['label_description'] = label_descriptions
    train_data['label_score'] = label_scores
    
    
    vertex_xs = []
    vertex_ys = []
    bounding_confidences = []
    bounding_importance_fracs = []
    dominant_blues = []
    dominant_greens = []
    dominant_reds = []
    dominant_pixel_fracs = []
    dominant_scores = []
    label_descriptions = []
    label_scores = []
    nf_count = 0
    nl_count = 0
    for pet in test_data['PetID'].values:
        try:
            with open('../input/petfinder-adoption-prediction/test_metadata/' + pet + '-1.json', 'r') as f:
                data = json.load(f)
            vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
            vertex_xs.append(vertex_x)
            vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
            vertex_ys.append(vertex_y)
            bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
            bounding_confidences.append(bounding_confidence)
            bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
            bounding_importance_fracs.append(bounding_importance_frac)
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('blue',-1)
            dominant_blues.append(dominant_blue)
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('green',-1)
            dominant_greens.append(dominant_green)
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('red',-1)
            dominant_reds.append(dominant_red)
            dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
            dominant_pixel_fracs.append(dominant_pixel_frac)
            dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
            dominant_scores.append(dominant_score)
            if data.get('labelAnnotations'):
                label_description = data['labelAnnotations'][0]['description']
                label_descriptions.append(label_description)
                label_score = data['labelAnnotations'][0]['score']
                label_scores.append(label_score)
            else:
                nl_count += 1
                label_descriptions.append('nothing')
                label_scores.append(-1)
        except FileNotFoundError:
            nf_count += 1
            vertex_xs.append(-1)
            vertex_ys.append(-1)
            bounding_confidences.append(-1)
            bounding_importance_fracs.append(-1)
            dominant_blues.append(-1)
            dominant_greens.append(-1)
            dominant_reds.append(-1)
            dominant_pixel_fracs.append(-1)
            dominant_scores.append(-1)
            label_descriptions.append('nothing')
            label_scores.append(-1)
    
    print(nf_count)
    test_data[ 'vertex_x'] = vertex_xs
    test_data['vertex_y'] = vertex_ys
    test_data['bounding_confidence'] = bounding_confidences
    test_data['bounding_importance'] = bounding_importance_fracs
    test_data['dominant_blue'] = dominant_blues
    test_data['dominant_green'] = dominant_greens
    test_data['dominant_red'] = dominant_reds
    test_data['dominant_pixel_frac'] = dominant_pixel_fracs
    test_data['dominant_score'] = dominant_scores
    test_data['label_description'] = label_descriptions
    test_data['label_score'] = label_scores
    
    del  vertex_xs,vertex_ys,bounding_confidences,bounding_importance_fracs,dominant_blues,dominant_greens,dominant_reds,dominant_pixel_fracs,dominant_scores
    del label_descriptions,label_scores
    gc.collect()
    
    train_data['label_description'] =train_data['label_description'].astype(np.str)
    train_data['label_description'] =train_data['label_description'].astype('category')
    
    test_data['label_description'] =test_data['label_description'].astype(np.str)
    test_data['label_description'] =test_data['label_description'].astype('category')
    
    tfv = TfidfVectorizer(min_df=3,  max_features=10000,
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
        stop_words = 'english')
    print("TFIDF....")
    tfv.fit(list(train_data['Description'].values)+list(test_data['Description'].values))
    X =  tfv.transform(train_data['Description'])
    X_test = tfv.transform(test_data['Description'])
    
    svd = TruncatedSVD(n_components=10)
    svd.fit(X)
    print(svd.explained_variance_ratio_.sum())
    print(svd.explained_variance_ratio_)
    X_svg = svd.transform(X)
    X_svg = pd.DataFrame(X_svg, columns=['svg_{}'.format(i) for i in range(10)])
    
    X_test_svg = svd.transform(X_test)
    X_test_svg = pd.DataFrame(X_test_svg, columns=['svg_{}'.format(i) for i in range(10)])
    train_data = pd.concat((train_data, X_svg), axis=1)
    test_data = pd.concat((test_data, X_test_svg), axis=1)
    
    breed=pd.read_csv("../input/petfinder-adoption-prediction/breed_labels.csv")
    color = pd.read_csv("../input/petfinder-adoption-prediction/color_labels.csv")
    color_dict = dict(zip(color['ColorID'].values.astype("str"),color['ColorName'].values))
    breed_dict = dict(zip(breed['BreedID'].values.astype("str"),breed['BreedName'].values))
    def get_text(df):
        x=""
        if df['Type']==1:
            x+="dog"+" "
        if df['Type']==2:
            x+="cat"+" "
        for i in ['Breed1',"Breed2"]:
            if df[i]==0:
                continue
            x+=breed_dict[str(df[i])]+" "
        for i in ["Color1","Color2","Color3"]:
            if df[i]==0:
                continue
            x+=color_dict[str(df[i])]+" "
        x=x+df['Description']
        return x
    train_data['concat_text']=train_data.apply(lambda x:get_text(x),1)
    test_data['concat_text']=test_data.apply(lambda x:get_text(x),1)
    
    train_desc=train_data['concat_text'].values
    test_desc=test_data['concat_text'].values
    
    tfv.fit(list(train_data['concat_text'].values)+list(test_data['concat_text'].values))
    X =  tfv.transform(train_data['concat_text'])
    X_test = tfv.transform(test_data['concat_text'])
    
    

    svd = NMF(n_components=5,random_state=100)
    svd.fit(vstack([X,X_test]))
    X_svg = svd.transform(X)
    X_svg = pd.DataFrame(X_svg, columns=['nmf_{}'.format(i) for i in range(5)])
    
    X_test_svg = svd.transform(X_test)
    X_test_svg = pd.DataFrame(X_test_svg, columns=['nmf_{}'.format(i) for i in range(5)])
    train_data = pd.concat((train_data, X_svg), axis=1)
    test_data = pd.concat((test_data, X_test_svg), axis=1)
    
    svd = LatentDirichletAllocation(n_components=5,max_iter=30, random_state=100)
    svd.fit(vstack([X,X_test]))
    X_svg = svd.transform(X)
    X_svg = pd.DataFrame(X_svg, columns=['lda_{}'.format(i) for i in range(5)])
    
    X_test_svg = svd.transform(X_test)
    X_test_svg = pd.DataFrame(X_test_svg, columns=['lda_{}'.format(i) for i in range(5)])
    train_data = pd.concat((train_data, X_svg), axis=1)
    test_data = pd.concat((test_data, X_test_svg), axis=1)
    
    tfv =  CountVectorizer(min_df=3,  
        token_pattern=r'\w{1,}',
        ngram_range=(1, 5),
        stop_words = 'english')
    tfv.fit(list(train_data['Description'].values)+list(test_data['Description'].values))
    X =  tfv.transform(train_data['Description'])
    X_test = tfv.transform(test_data['Description'])
    
    svd = TruncatedSVD(n_components=5)
    svd.fit(vstack([X,X_test]))
    print(svd.explained_variance_ratio_.sum())
    print(svd.explained_variance_ratio_)
    X_svg = svd.transform(X)
    X_svg = pd.DataFrame(X_svg, columns=['nb_{}'.format(i) for i in range(5)])
    
    X_test_svg = svd.transform(X_test)
    X_test_svg = pd.DataFrame(X_test_svg, columns=['nb_{}'.format(i) for i in range(5)])
    train_data = pd.concat((train_data, X_svg), axis=1)
    test_data = pd.concat((test_data, X_test_svg), axis=1)
    
    svd = LatentDirichletAllocation(n_components=5,max_iter=30, random_state=10)
    svd.fit(vstack([X,X_test]))
    X_svg = svd.transform(X)
    X_svg = pd.DataFrame(X_svg, columns=['c_lda_{}'.format(i) for i in range(5)])
    
    X_test_svg = svd.transform(X_test)
    X_test_svg = pd.DataFrame(X_test_svg, columns=['c_lda_{}'.format(i) for i in range(5)])
    train_data = pd.concat((train_data, X_svg), axis=1)
    test_data = pd.concat((test_data, X_test_svg), axis=1)
    
    onehot_col=["Breed1","Breed2","Color1","Color2","Color3",'State','Gender','MaturitySize','MaturitySize','FurLength','Vaccinated',
           'Dewormed','Sterilized','Health']
    data=pd.concat([train_data,test_data])
    data.index=range(len(data))
    onehot_df=pd.DataFrame(index=range(len(data)))
    for i in onehot_col:
        temp=pd.get_dummies(data[i],prefix=i)
        onehot_df=pd.concat([onehot_df,temp],1)
        
    svd = TruncatedSVD(n_components=5)
    svd.fit(onehot_df)
    print(svd.explained_variance_ratio_.sum())
    print(svd.explained_variance_ratio_)
    oh_df = svd.transform(onehot_df)
    oh_df = pd.DataFrame(oh_df, columns=['oh_{}'.format(i) for i in range(5)])
    oh_df['PetID']=data['PetID']
    
    del X,X_test,onehot_df,data
    gc.collect()
    
    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
                                                              "hash_ngrams_weights": [1.5, 1.0],
                                                              "hash_size": 2 ** 29,
                                                              "norm": None,
                                                              "tf": 'binary',
                                                              "idf": None,
                                                              }), procs=8)
    x_train = wb.fit_transform(train_data["Description"])
    x_test = wb.transform(test_data["Description"])
    mask = np.array(np.clip(x_train.getnnz(axis=0) -8 , 0, 1), dtype=bool)
    x_train=x_train[:,mask]
    x_test=x_test[:,mask]
    print(x_test.shape)
    
    svd = TruncatedSVD(n_components=5)
    svd.fit(x_train)
    print(svd.explained_variance_ratio_.sum())
    print(svd.explained_variance_ratio_)
    X_svg = svd.transform(x_train)
    X_svg = pd.DataFrame(X_svg, columns=['wb_{}'.format(i) for i in range(5)])
    
    X_test_svg = svd.transform(x_test)
    X_test_svg = pd.DataFrame(X_test_svg, columns=['wb_{}'.format(i) for i in range(5)])
    train_data = pd.concat((train_data, X_svg), axis=1)
    test_data = pd.concat((test_data, X_test_svg), axis=1)
    del x_test,x_train,X_test_svg,X_svg,wb,svd
    gc.collect()
    
    train_df = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
    img_size = 256
    batch_size = 16
    pet_ids = train_df['PetID'].values
    n_batches = len(pet_ids) // batch_size + 1
    
    extract_model = img_model()
    img_features = {}
    for b in tqdm_notebook(range(n_batches)):
        start = b*batch_size
        end = (b+1)*batch_size
        batch_pets = pet_ids[start:end]
        batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
        for i,pet_id in enumerate(batch_pets):
            try:
                batch_images[i] = load_image("../input/petfinder-adoption-prediction/train_images/", pet_id)
            except:
                pass
        batch_preds = extract_model.predict(batch_images)
        for i,pet_id in enumerate(batch_pets):
            img_features[pet_id] = batch_preds[i]
    train_img_feat = pd.DataFrame.from_dict(img_features, orient='index')
    #train_feats.to_csv('train_img_features.csv')
    
    test_df = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
    pet_ids = test_df['PetID'].values
    n_batches = len(pet_ids) // batch_size + 1
    
    img_features = {}
    for b in tqdm_notebook(range(n_batches)):
        start = b*batch_size
        end = (b+1)*batch_size
        batch_pets = pet_ids[start:end]
        batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
        for i,pet_id in enumerate(batch_pets):
            try:
                batch_images[i] = load_image("../input/petfinder-adoption-prediction/test_images/", pet_id)
            except:
                pass
        batch_preds = extract_model.predict(batch_images)
        for i,pet_id in enumerate(batch_pets):
            img_features[pet_id] = batch_preds[i]
            
    test_img_feat = pd.DataFrame.from_dict(img_features, orient='index')
    del img_features
    gc.collect()
    
    pca = PCA(n_components=5)
    pca.fit(train_img_feat.values)
    train_img=pca.transform(train_img_feat.values)
    test_img=pca.transform(test_img_feat.values)
    train_img = pd.DataFrame(train_img, columns=['pca_{}'.format(i) for i in range(5)])
    test_img = pd.DataFrame(test_img, columns=['pca_{}'.format(i) for i in range(5)])
    train_data = pd.concat((train_data, train_img), axis=1)
    test_data = pd.concat((test_data, test_img), axis=1)
    del train_img,test_img,pca
    gc.collect()
    
    state= pd.read_csv("../input/petfinder-adoption-prediction/state_labels.csv")

    state_dict={
        "Johor":[3233434,19210,168],
        "Kedah":[1890098,9500,199],
        "Kelantan":[1459994,15099,97],
        "Kuala Lumpur":[1627172,243,6696],
        "Labuan":[85272,91,937],
        "Melaka":[788706,1664,474],
        "Negeri Sembilan":[997071,6686,149],
        "Pahang":[1443365,36137,40],
        "Perak":[2258428,21035,107],
        "Perlis":[227025,821,277],
        "Pulau Pinang":[1520143,1048,1451],
        "Sabah":[3120040,73631,42],
        "Sarawak":[2420009,124450,19],
        "Selangor":[5411324,8104,668],
        "Terengganu":[1015776,13035,78]
    }
    def get_state_feat(df):
        df['state_people/area_ratio']=state_dict[df['StateName']][2]
        return df
    state=state.apply(lambda x:get_state_feat(x),1)
    state['state_rank']=state['state_people/area_ratio'].rank()
    state_ratio_dict=dict(zip(state.StateID.values,state['state_people/area_ratio'].values))
    state_rank_dict=dict(zip(state.StateID.values,state['state_rank'].values))
    
    train_data['State_ratio']=train_data['State'].map(state_ratio_dict)
    test_data['State_ratio']=test_data['State'].map(state_ratio_dict)
    
    train_data['State_rank']=train_data['State'].map(state_rank_dict)
    test_data['State_rank']=test_data['State'].map(state_rank_dict)
    
    del state_rank_dict,state_ratio_dict,state
    gc.collect()
    # train_data['is_high_state_ratio']=train_data['State_ratio'].apply(lambda x:1 if x>168 else 0,1)
    # test_data['is_high_state_ratio']=test_data['State_ratio'].apply(lambda x:1 if x>168 else 0,1)
    
    def get_breed(df):
        x=""
        for i in ["Breed1","Breed2"]:
            if df[i]==0:
                continue
            x+=breed_dict[str(df[i])]+" "
        return x
    train_data['breed']=train_data.apply(lambda x:get_breed(x),1)
    test_data['breed']=test_data.apply(lambda x:get_breed(x),1)   
    
    train_data['Breed1']=train_data['Breed1'].astype("str")
    train_data['Breed1']=train_data['Breed1'].map(breed_dict)
    train_data['Breed1']=train_data['Breed1'].replace(np.nan,"null")
    train_data['Breed2']=train_data['Breed2'].astype("str")
    train_data['Breed2']=train_data['Breed2'].map(breed_dict)
    train_data['Breed2']=train_data['Breed2'].replace(np.nan,"null")
    
    test_data['Breed1']=test_data['Breed1'].astype("str")
    test_data['Breed1']=test_data['Breed1'].map(breed_dict)
    test_data['Breed1']=test_data['Breed1'].replace(np.nan,"null")
    test_data['Breed2']=test_data['Breed2'].astype("str")
    test_data['Breed2']=test_data['Breed2'].map(breed_dict)
    test_data['Breed2']=test_data['Breed2'].replace(np.nan,"null")
    
    def get_color_cnt(df):
        color_list=[]
        for i in ["Color1","Color2","Color3"]:
            if df[i]!=0:
                color_list.append(df[i])
        return len(set(color_list))
    train_data['color_cnt']=train_data.apply(lambda x:get_color_cnt(x),1)
    test_data['color_cnt']=test_data.apply(lambda x:get_color_cnt(x),1)
    
    def get_color(df):
        x=""
        for i in ["Color1","Color2","Color3"]:
            if df[i]==0:
                continue
            x+=color_dict[str(df[i])]+" "
        return x
    train_data['color']=train_data.apply(lambda x:get_color(x),1)
    test_data['color']=test_data.apply(lambda x:get_color(x),1)
    
    mean_encoder = MeanEncoder( categorical_features=['Breed1', 'breed'],target_type ='regression')
    train_data = mean_encoder.fit_transform(train_data, train_data['AdoptionSpeed'])
    test_data = mean_encoder.transform(test_data)
    for col in ['Breed1',"Breed2","color","breed"]:
        lbl = LabelEncoder()
        train_data[col]=train_data[col].fillna(0)
        test_data[col]=test_data[col].fillna(0)
        lbl.fit(list(train_data[col].values)+list(test_data[col].values))
        train_data[col]=lbl.transform(train_data[col])
        test_data[col]=lbl.transform(test_data[col])
        
    
    cols = [x for x in train_data.columns if x not in ['Breed1',"breed",'label_description',"color","Breed2","State","lan_type","malai_type","Type","concat_text","is_group","Name",'PetID',"Description",'AdoptionSpeed']]
    data=pd.concat([train_data,test_data])
    train_data[cols]=train_data[cols].fillna(0)
    test_data[cols]=test_data[cols].fillna(0)
    ############################ 切分数据集 ##########################
    print('开始进行一些前期处理')
    train_feature = train_data[cols].values
    test_feature = test_data[cols].values
        # 五则交叉验证
    n_folds = 5
    print('处理完毕')
    df_stack2 = pd.DataFrame()
    df_stack2['PetID']=data['PetID']
    for label in ["AdoptionSpeed"]:
        score = train_data[label]
        
       
        ########################### SGD(随机梯度下降) ################################
        print('sgd stacking')
        stack_train = np.zeros((len(train_data),1))
        stack_test = np.zeros((len(test_data),1))
        score_va = 0
    
        sk = StratifiedKFold( n_splits=5, random_state=1017)
        for i, (tr, va) in enumerate(sk.split(train_feature, score)):
            print('stack:%d/%d' % ((i + 1), n_folds))
            sgd = SGDRegressor(random_state=1017,)
            sgd.fit(train_feature[tr], score[tr])
            score_va = sgd.predict(train_feature[va])
            score_te = sgd.predict(test_feature)
            print('得分' + str(mean_squared_error(score[va], sgd.predict(train_feature[va]))))
            stack_train[va,0] = score_va
            stack_test[:,0]+= score_te
        stack_test /= n_folds
        stack = np.vstack([stack_train, stack_test])
        df_stack2['tfidf_sgd_classfiy_{}'.format(label)] = stack[:,0]
    
    
        ########################### pac(PassiveAggressiveClassifier) ################################
        print('PAC stacking')
        stack_train = np.zeros((len(train_data),1))
        stack_test = np.zeros((len(test_data),1))
        score_va = 0
    
        sk = StratifiedKFold( n_splits=5, random_state=1017)
        for i, (tr, va) in enumerate(sk.split(train_feature, score)):
            print('stack:%d/%d' % ((i + 1), n_folds))
            pac = PassiveAggressiveRegressor(random_state=1017)
            pac.fit(train_feature[tr], score[tr])
            score_va = pac.predict(train_feature[va])
            score_te = pac.predict(test_feature)
          
            print('得分' + str(mean_squared_error(score[va], pac.predict(train_feature[va]))))
            stack_train[va,0] = score_va
            stack_test[:,0] += score_te
        stack_test /= n_folds
        stack = np.vstack([stack_train, stack_test])
    
        df_stack2['tfidf_pac_classfiy_{}'.format(label)] = stack[:,0]
        
    
    
        
    
        ########################### FTRL ################################
        print('MultinomialNB stacking')
        stack_train = np.zeros((len(train_data),1))
        stack_test = np.zeros((len(test_data),1))
        score_va = 0
    
        sk = StratifiedKFold( n_splits=5, random_state=1017)
        for i, (tr, va) in enumerate(sk.split(train_feature, score)):
            print('stack:%d/%d' % ((i + 1), n_folds))
            clf = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=train_feature.shape[1], iters=50, inv_link="identity", threads=1)
            clf.fit(train_feature[tr], score[tr])
            score_va = clf.predict(train_feature[va])
            score_te = clf.predict(test_feature)
          
            print('得分' + str(mean_squared_error(score[va], clf.predict(train_feature[va]))))
            stack_train[va,0] = score_va
            stack_test[:,0] += score_te
        stack_test /= n_folds
        stack = np.vstack([stack_train, stack_test])
        
        df_stack2['tfidf_FTRL_classfiy_{}'.format(label)] = stack[:,0]
        
        ########################### ridge(RidgeClassfiy) ################################
        print('RidgeClassfiy stacking')
        stack_train = np.zeros((len(train_data),1))
        stack_test = np.zeros((len(test_data),1))
        score_va = 0
    
        sk = StratifiedKFold( n_splits=5, random_state=1017)
        for i, (tr, va) in enumerate(sk.split(train_feature, score)):
            print('stack:%d/%d' % ((i + 1), n_folds))
            ridge = Ridge(solver="sag", fit_intercept=True, random_state=42, alpha=30) 
            ridge.fit(train_feature[tr], score[tr])
            score_va = ridge.predict(train_feature[va])
            score_te = ridge.predict(test_feature)
           
            print('得分' + str(mean_squared_error(score[va], ridge.predict(train_feature[va]))))
            stack_train[va,0] = score_va
            stack_test[:,0] += score_te
        stack_test /= n_folds
        stack = np.vstack([stack_train, stack_test])
    
        df_stack2['tfidf_ridge_classfiy_{}'.format(label)] = stack[:,0]
        
        ############################ Linersvc(LinerSVC) ################################
        print('LinerSVC stacking')
        stack_train = np.zeros((len(train_data),1))
        stack_test = np.zeros((len(test_data),1))
        score_va = 0
    
        sk = StratifiedKFold( n_splits=5, random_state=1017)
        for i, (tr, va) in enumerate(sk.split(train_feature, score)):
            print('stack:%d/%d' % ((i + 1), n_folds))
            lsvc = LinearSVR(random_state=1017)
            lsvc.fit(train_feature[tr], score[tr])
            score_va = lsvc.predict(train_feature[va])
            score_te = lsvc.predict(test_feature)
           
            print('得分' + str(mean_squared_error(score[va], lsvc.predict(train_feature[va]))))
            stack_train[va,0] = score_va
            stack_test[:,0] += score_te
        stack_test /= n_folds
        stack = np.vstack([stack_train, stack_test])
    
        df_stack2['tfidf_lsvc_classfiy_{}'.format(label)] = stack[:,0]
        
    # df_stack.to_csv('graph_tfidf_classfiy.csv', index=None, encoding='utf8')
    print('tfidf特征已保存\n')
    del stack,train_feature,test_feature,stack_train, stack_test,data
    gc.collect()
    
    
    
    train=pd.merge(train_data,oh_df,on="PetID",how="left")
    test=pd.merge(test_data,oh_df,on="PetID",how="left")
    
    train=pd.merge(train,df_stack2,on="PetID",how="left")
    test=pd.merge(test,df_stack2,on="PetID",how="left")
    
    del train_data,test_data,df_stack2,oh_df
    gc.collect()
    
    return train,test,train_desc,test_desc,train_img_feat,test_img_feat
    
def runLGB_c(train_X, train_y, test_X, test_y, test_X2, params):
#     print('Prep LGB')
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
#     print('Train LGB')
    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    early_stop = None
    if params.get('early_stop'):
        early_stop = params.pop('early_stop')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
#                       fobj=softkappaObj,
                      verbose_eval=verbose_eval,
#                       feval=kappa_scorer,
                      early_stopping_rounds=early_stop)
    print('Predict 1/2')
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    log=log_loss(test_y,pred_test_y)
    print("log_loss:",log)
    class_list=[0,1,2,3,4]
    pred_test_y=np.array([sum(pred_test_y[ix]*class_list) for
                               ix in range(len(pred_test_y[:,0]))]) 
    optR = OptimizedRounder()
    optR.fit(pred_test_y, test_y)
    len_0 = sum([1 for i in test_y if i==0])
    coefficients = optR.coefficients()
    pred_test_y_k = optR.predict(pred_test_y, coefficients,len_0)
   
    print("Valid Counts = ", Counter(test_y))
    print("Predicted Counts = ", Counter(pred_test_y_k))
    print("Coefficients = ", coefficients)
    qwk = cohen_kappa_score(test_y, pred_test_y_k,weights='quadratic')
    print("QWK = ", qwk)
    print('Predict 2/2')
    pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
    pred_test_y2=np.array([sum(pred_test_y2[ix]*class_list) for
                               ix in range(len(pred_test_y2[:,0]))]) 
   
    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), model.feature_importance(), coefficients, qwk,log
train,test,train_desc,test_desc,train_img_feat,test_img_feat=get_feat1()
train1=train.copy()
test1=test.copy()
train1.drop(['pca_{}'.format(i) for i in range(5)],1,inplace=True)
test1.drop(['pca_{}'.format(i) for i in range(5)],1,inplace=True)

features = [x for x in train.columns if x not in ['Breed1',"breed","color","Breed2","State","lan_type","malai_type","Type","concat_text","is_group","Name",'PetID',"Description",'AdoptionSpeed']]

label='AdoptionSpeed'
###model 1
# params = {
# #     'application': 'regression',
#     'objective': 'multiclass', 
#     "num_class":5,
#           'boosting': 'gbdt',
# #           'metric': 'rmse',
#     'metric':{'multi_logloss',},
#           'num_leaves': 80,
#          'max_depth':9,
#           'learning_rate': 0.01,
#           'bagging_fraction': 0.90,
#            "bagging_freq":3,
#           'feature_fraction': 0.85,
#           'min_split_gain': 0.01,
#           'min_child_samples': 150,
#           "lambda_l1": 0.1,
#           'verbosity': -1,
#           'early_stop': 100,
#           'verbose_eval': 200,
#           "data_random_seed":3,
# #           "random_state":1017,
#           'num_rounds': 10000}
params = {
#     'application': 'regression',
    'objective': 'multiclass', 
    "num_class":5,
          'boosting': 'gbdt',
#           'metric': 'rmse',
    'metric':{'multi_logloss',},
          'num_leaves': 55,
         'max_depth':9,
        'max_bin': 45,
          'learning_rate': 0.01,
          'bagging_fraction': 0.9879639408647978,
            "bagging_freq":41,
           'feature_fraction': 0.5849356442713105,
           'min_split_gain': 0.6118528947223795,
         'min_child_samples': 83,
     'min_child_weight': 0.2912291401980419,
     'lambda_l1': 0.18182496720710062,
          'lambda_l2': 0.18340450985343382,
          'verbosity': -1,
          'early_stop': 100,
          'verbose_eval': 200,
          "data_random_seed":17,
#           "random_state":1017,
          'num_rounds': 10000}
results = run_cv_model(train[features], test[features], train[label], runLGB_c, params, rmse, 'LGB')    
imports = results['importance'].groupby('feature')['feature', 'importance'].mean().reset_index()
imp=imports.sort_values('importance', ascending=False)
print(imp)
lgb1_train=[r[0] for r in results['train']]
lgb1_test=[r[0] for r in results['test']]
t1=time.time()
print("model1 cost:{} s".format(t1-start_time))

###model 2
features = [x for x in train.columns if x not in ['label_description','Breed1',"breed","color","Breed2","State","lan_type","malai_type","Type","concat_text","is_group","Name",'PetID',"Description",'AdoptionSpeed']]
def runCAT(train_X, train_y, test_X, test_y, test_X2, params):
#     print('Prep LGB')
    d_train = Pool(train_X, label=train_y)
    d_valid = Pool(test_X, label=test_y)
    watchlist = (d_train, d_valid)
#     print('Train LGB')
    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    early_stop = None
    if params.get('early_stop'):
        early_stop = params.pop('early_stop')
    model = CatBoostClassifier(iterations=num_rounds, 
        learning_rate = 0.03,
        od_type='Iter',
         od_wait=early_stop,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        bagging_temperature=0.9,                   
        random_seed = 2019,
        task_type='GPU'
                          )
    model.fit(d_train,eval_set=d_valid,
            use_best_model=True,
            verbose=verbose_eval
                         )
    
    print('Predict 1/2')
    pred_test_y = model.predict_proba(test_X)
    log=log_loss(test_y,pred_test_y)
    print("log_loss:",log)
    class_list=[0,1,2,3,4]
    pred_test_y=np.array([sum(pred_test_y[ix]*class_list) for
                               ix in range(len(pred_test_y[:,0]))]) 
    optR = OptimizedRounder()
    optR.fit(pred_test_y, test_y)
    len_0 = sum([1 for i in test_y if i==0])
    coefficients = optR.coefficients()
    pred_test_y_k = optR.predict(pred_test_y, coefficients,len_0)
   
    print("Valid Counts = ", Counter(test_y))
    print("Predicted Counts = ", Counter(pred_test_y_k))
    print("Coefficients = ", coefficients)
    qwk = cohen_kappa_score(test_y, pred_test_y_k,weights='quadratic')
    print("QWK = ", qwk)
    print('Predict 2/2')
    pred_test_y2 =  model.predict_proba(test_X2)
    pred_test_y2=np.array([sum(pred_test_y2[ix]*class_list) for
                               ix in range(len(pred_test_y2[:,0]))]) 
   
    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), 0, coefficients, qwk,log
results = run_cv_model(train[features], test[features], train[label], runCAT, params, rmse, 'CAT')
cat1_train=[r[0] for r in results['train']]
cat1_test=[r[0] for r in results['test']]
t2=time.time()
print("model2 cost:{} s".format(t2-t1))
###model 3
features = [x for x in train.columns if x not in ['Breed1_pred',"breed_pred","color","Breed2","State","lan_type","malai_type","Type","concat_text","is_group","Name",'PetID',"Description",'AdoptionSpeed']]
label='AdoptionSpeed'
params = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 80,
         'max_depth':9,
          'learning_rate': 0.01,
          'bagging_fraction': 0.9,
           "bagging_freq":3,
          'feature_fraction': 0.85,
          'min_split_gain': 0.01,
          'min_child_samples': 150,
          "lambda_l1": 0.1,
          'verbosity': -1,
          'early_stop': 100,
          'verbose_eval': 200,
           "data_random_seed":3,
#           "random_state":1017,
          'num_rounds': 10000}
def runLGB(train_X, train_y, test_X, test_y, test_X2, params):
#     print('Prep LGB')
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
#     print('Train LGB')
    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    early_stop = None
    if params.get('early_stop'):
        early_stop = params.pop('early_stop')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
#                       fobj=softkappaObj,
                      verbose_eval=verbose_eval,
#                       feval=kappa_scorer,
                      early_stopping_rounds=early_stop)
    print('Predict 1/2')
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    log=0#log_loss(test_y,pred_test_y)
    print("log_loss:",log)
    class_list=[0,1,2,3,4]
#     pred_test_y=np.array([sum(pred_test_y[ix]*class_list) for
#                                ix in range(len(pred_test_y[:,0]))]) 
    optR = OptimizedRounder()
    optR.fit(pred_test_y, test_y)
    len_0 = sum([1 for i in test_y if i==0])
    coefficients = optR.coefficients()
    pred_test_y_k = optR.predict(pred_test_y, coefficients,len_0)
    
    print("Valid Counts = ", Counter(test_y))
    print("Predicted Counts = ", Counter(pred_test_y_k))
    print("Coefficients = ", coefficients)
    qwk = cohen_kappa_score(test_y, pred_test_y_k,weights='quadratic')
    print("QWK = ", qwk)
    print('Predict 2/2')
    pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
#     pred_test_y2=np.array([sum(pred_test_y2[ix]*class_list) for
#                                ix in range(len(pred_test_y2[:,0]))]) 
   
    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), model.feature_importance(), coefficients, qwk,log
results = run_cv_model(train[features], test[features], train[label], runLGB, params, rmse, 'LGB')

lgb2_train=[r[0] for r in results['train']]
lgb2_test=[r[0] for r in results['test']]
del train,test
gc.collect()
t3=time.time()
print("model3 cost:{} s".format(t3-t2))
###model 4
def get_feat2():
    train_data = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")
    test_data = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")
    def get_purebreed_feat(df):
        if df['Breed2']==0 and df['Breed1']!=307:
            return 1
        return 0
    train_data['purebreed']=train_data.apply(lambda x:get_purebreed_feat(x),1)
    test_data['purebreed']=test_data.apply(lambda x:get_purebreed_feat(x),1)
    train_data['is_group']=train_data.Gender.apply(lambda x:1 if x==3 else 0,1)
    test_data['is_group']=test_data.Gender.apply(lambda x:1 if x==3 else 0,1)
    def name_deal(df):
        if "No Name" in df:
            return np.nan
        if df =="nan":
            return np.nan
        else:
            return df
    train_data['Name'] = train_data['Name'].apply(lambda x:name_deal(str(x)),1)
    test_data['Name'] = test_data['Name'].apply(lambda x:name_deal(str(x)),1)
    
    train_data['Name'],indexer=pd.factorize(train_data['Name'])
    test_data['Name'] = indexer.get_indexer(test_data['Name'])
    group = train_data['RescuerID'].values
    rescuer_df=train_data.groupby("RescuerID",as_index=False).count()[["RescuerID","PetID"]]
    rescuer_df.columns=["RescuerID","rescuer_cnt"]
    train_data=pd.merge(train_data,rescuer_df,on="RescuerID",how="left")
    train_data.drop("RescuerID",1,inplace=True)
    
    rescuer_df=test_data.groupby("RescuerID",as_index=False).count()[["RescuerID","PetID"]]
    rescuer_df.columns=["RescuerID","rescuer_cnt"]
    test_data=pd.merge(test_data,rescuer_df,on="RescuerID",how="left")
    test_data.drop("RescuerID",1,inplace=True)
    
    eng_stopwords = set(stopwords.words("english"))

    ## Number of words in the text ##
    train_data["num_words"] = train_data["Description"].apply(lambda x: len(str(x).split()))
    test_data["num_words"] = test_data["Description"].apply(lambda x: len(str(x).split()))
    
    ## Number of unique words in the text ##
    train_data["num_unique_words"] = train_data["Description"].apply(lambda x: len(set(str(x).split())))
    test_data["num_unique_words"] = test_data["Description"].apply(lambda x: len(set(str(x).split())))
    
    ## Number of characters in the text ##
    train_data["num_chars"] = train_data["Description"].apply(lambda x: len(str(x)))
    test_data["num_chars"] = test_data["Description"].apply(lambda x: len(str(x)))
    
    ## Number of stopwords in the text ##
    train_data["num_stopwords"] = train_data["Description"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
    test_data["num_stopwords"] = test_data["Description"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
    
    ## Number of punctuations in the text ##
    train_data["num_punctuations"] =train_data['Description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
    test_data["num_punctuations"] =test_data['Description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
    
    ## Number of title case words in the text ##
    train_data["num_words_upper"] = train_data["Description"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    test_data["num_words_upper"] = test_data["Description"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    
    ## Number of title case words in the text ##
    train_data["num_words_title"] = train_data["Description"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    test_data["num_words_title"] = test_data["Description"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    
    ## Average length of the words in the text ##
    train_data["mean_word_len"] = train_data["Description"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    test_data["mean_word_len"] = test_data["Description"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    
    train_data['Description'] = train_data['Description'].fillna("null")
    test_data['Description'] = test_data['Description'].fillna("null")
    data = pd.concat([train_data,test_data])
    data.index=range(len(data))
    data_id=data['PetID'].values
    
    
    doc_sent_mag = []
    doc_sent_score = []
    nf_count = 0
    for pet in train_data['PetID'].values:
        try:
            with open('../input/petfinder-adoption-prediction/train_sentiment/' + pet + '.json', 'r') as f:
                sentiment = json.load(f)
            doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
            doc_sent_score.append(sentiment['documentSentiment']['score'])
        except FileNotFoundError:
            nf_count += 1
            doc_sent_mag.append(-1)
            doc_sent_score.append(-1)
    
    train_data['doc_sent_mag'] = doc_sent_mag
    train_data['doc_sent_score'] = doc_sent_score
    
    doc_sent_mag = []
    doc_sent_score = []
    nf_count = 0
    for pet in test_data['PetID'].values:
        try:
            with open('../input/petfinder-adoption-prediction/test_sentiment/' + pet + '.json', 'r') as f:
                sentiment = json.load(f)
            doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
            doc_sent_score.append(sentiment['documentSentiment']['score'])
        except FileNotFoundError:
            nf_count += 1
            doc_sent_mag.append(-1)
            doc_sent_score.append(-1)
    
    test_data['doc_sent_mag'] = doc_sent_mag
    test_data['doc_sent_score'] = doc_sent_score
    
    vertex_xs = []
    vertex_ys = []
    bounding_confidences = []
    bounding_importance_fracs = []
    dominant_blues = []
    dominant_greens = []
    dominant_reds = []
    dominant_pixel_fracs = []
    dominant_scores = []
    label_descriptions = []
    label_scores = []
    nf_count = 0
    nl_count = 0
    for pet in train_data['PetID'].values:
        try:
            with open('../input/petfinder-adoption-prediction/train_metadata/' + pet + '-1.json', 'r') as f:
                data = json.load(f)
            vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
            vertex_xs.append(vertex_x)
            vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
            vertex_ys.append(vertex_y)
            bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
            bounding_confidences.append(bounding_confidence)
            bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
            bounding_importance_fracs.append(bounding_importance_frac)
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('blue',-1)
            dominant_blues.append(dominant_blue)
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('green',-1)
            dominant_greens.append(dominant_green)
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('red',-1)
            dominant_reds.append(dominant_red)
            dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
            dominant_pixel_fracs.append(dominant_pixel_frac)
            dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
            dominant_scores.append(dominant_score)
            if data.get('labelAnnotations'):
                label_description = data['labelAnnotations'][0]['description']
                label_descriptions.append(label_description)
                label_score = data['labelAnnotations'][0]['score']
                label_scores.append(label_score)
            else:
                nl_count += 1
                label_descriptions.append('nothing')
                label_scores.append(-1)
        except FileNotFoundError:
            nf_count += 1
            vertex_xs.append(-1)
            vertex_ys.append(-1)
            bounding_confidences.append(-1)
            bounding_importance_fracs.append(-1)
            dominant_blues.append(-1)
            dominant_greens.append(-1)
            dominant_reds.append(-1)
            dominant_pixel_fracs.append(-1)
            dominant_scores.append(-1)
            label_descriptions.append('nothing')
            label_scores.append(-1)
    
    print(nf_count)
    print(nl_count)
    train_data[ 'vertex_x'] = vertex_xs
    train_data['vertex_y'] = vertex_ys
    train_data['bounding_confidence'] = bounding_confidences
    train_data['bounding_importance'] = bounding_importance_fracs
    train_data['dominant_blue'] = dominant_blues
    train_data['dominant_green'] = dominant_greens
    train_data['dominant_red'] = dominant_reds
    train_data['dominant_pixel_frac'] = dominant_pixel_fracs
    train_data['dominant_score'] = dominant_scores
    train_data['label_description'] = label_descriptions
    train_data['label_score'] = label_scores
    
    
    vertex_xs = []
    vertex_ys = []
    bounding_confidences = []
    bounding_importance_fracs = []
    dominant_blues = []
    dominant_greens = []
    dominant_reds = []
    dominant_pixel_fracs = []
    dominant_scores = []
    label_descriptions = []
    label_scores = []
    nf_count = 0
    nl_count = 0
    for pet in test_data['PetID'].values:
        try:
            with open('../input/petfinder-adoption-prediction/test_metadata/' + pet + '-1.json', 'r') as f:
                data = json.load(f)
            vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
            vertex_xs.append(vertex_x)
            vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
            vertex_ys.append(vertex_y)
            bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
            bounding_confidences.append(bounding_confidence)
            bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
            bounding_importance_fracs.append(bounding_importance_frac)
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('blue',-1)
            dominant_blues.append(dominant_blue)
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('green',-1)
            dominant_greens.append(dominant_green)
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('red',-1)
            dominant_reds.append(dominant_red)
            dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
            dominant_pixel_fracs.append(dominant_pixel_frac)
            dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
            dominant_scores.append(dominant_score)
            if data.get('labelAnnotations'):
                label_description = data['labelAnnotations'][0]['description']
                label_descriptions.append(label_description)
                label_score = data['labelAnnotations'][0]['score']
                label_scores.append(label_score)
            else:
                nl_count += 1
                label_descriptions.append('nothing')
                label_scores.append(-1)
        except FileNotFoundError:
            nf_count += 1
            vertex_xs.append(-1)
            vertex_ys.append(-1)
            bounding_confidences.append(-1)
            bounding_importance_fracs.append(-1)
            dominant_blues.append(-1)
            dominant_greens.append(-1)
            dominant_reds.append(-1)
            dominant_pixel_fracs.append(-1)
            dominant_scores.append(-1)
            label_descriptions.append('nothing')
            label_scores.append(-1)
    
    print(nf_count)
    test_data[ 'vertex_x'] = vertex_xs
    test_data['vertex_y'] = vertex_ys
    test_data['bounding_confidence'] = bounding_confidences
    test_data['bounding_importance'] = bounding_importance_fracs
    test_data['dominant_blue'] = dominant_blues
    test_data['dominant_green'] = dominant_greens
    test_data['dominant_red'] = dominant_reds
    test_data['dominant_pixel_frac'] = dominant_pixel_fracs
    test_data['dominant_score'] = dominant_scores
    test_data['label_description'] = label_descriptions
    test_data['label_score'] = label_scores
    
    del  vertex_xs,vertex_ys,bounding_confidences,bounding_importance_fracs,dominant_blues,dominant_greens,dominant_reds,dominant_pixel_fracs,dominant_scores
    del label_descriptions,label_scores,doc_sent_mag,doc_sent_score
    gc.collect()
    
    train_data['label_description'] =train_data['label_description'].astype(np.str)
    train_data['label_description'] =train_data['label_description'].astype('category')
    
    test_data['label_description'] =test_data['label_description'].astype(np.str)
    test_data['label_description'] =test_data['label_description'].astype('category')
    
    tfv = TfidfVectorizer(min_df=3,  max_features=10000,
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
        stop_words = 'english')
    print("TFIDF....")
    tfv.fit(list(train_data['Description'].values)+list(test_data['Description'].values))
    X =  tfv.transform(train_data['Description'])
    X_test = tfv.transform(test_data['Description'])
    
    svd = TruncatedSVD(n_components=120)
    svd.fit(X)
    print(svd.explained_variance_ratio_.sum())
    print(svd.explained_variance_ratio_)
    X = svd.transform(X)
    X = pd.DataFrame(X, columns=['nb_{}'.format(i) for i in range(120)])
    
    X_test = svd.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=['nb_{}'.format(i) for i in range(120)])
    train_data = pd.concat((train_data, X), axis=1)
    test_data = pd.concat((test_data, X_test), axis=1)
    
    del X,X_test
    gc.collect()
    return train_data,test_data
    
    
train,test=get_feat2()
train2=train.copy()
test2=test.copy()

features = [x for x in train.columns if x not in ["is_group","Name",'PetID',"Description",'AdoptionSpeed']]
label='AdoptionSpeed'
params = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 80,
#          'max_depth':11,
          'learning_rate': 0.01,
          'bagging_fraction': 0.9,
           "bagging_freq":3,
          'feature_fraction': 0.4,
          'min_split_gain': 0.01,
#           'min_child_samples': 30,
#           "lambda_l1": 0.1,
          'verbosity': -1,
          'early_stop': 100,
          'verbose_eval': 200,
#           "random_state":1017,
          'num_rounds': 10000}
results = run_cv_model(train[features], test[features], train[label], runLGB, params, rmse, 'LGB')
lgb3_train=[r[0] for r in results['train']]
lgb3_test=[r[0] for r in results['test']]
del train,test
gc.collect()
t4=time.time()
print("model4 cost:{} s".format(t4-t3))
def get_feat3():
    train_data = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")
    test_data = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")
    
    # petid列
    data = pd.concat([train_data, test_data], axis=0)
    del data['AdoptionSpeed']
    es = ft.EntitySet(id='data_id')
    es = es.entity_from_dataframe(entity_id='PetID', dataframe=data,
                                   index='PetID')
    
    need_deal_columns = ['Age', 'Breed1', 'Breed2', 'Color1', 'Color2', 'Color3', 'Description',
           'Dewormed', 'Fee', 'FurLength', 'Gender', 'Health', 'MaturitySize',
           'Name', 'PhotoAmt', 'Quantity', 'RescuerID', 'State',
           'Sterilized', 'Type', 'Vaccinated', 'VideoAmt']
    for i in need_deal_columns:
        data_RescuerID = pd.DataFrame()
        data_RescuerID[i] = list(data[i].unique())
        es = es.entity_from_dataframe(entity_id=i, dataframe=data_RescuerID,
                                   index=i)
        cr = ft.Relationship( es[i][i],
                        es['PetID'][i])
        es = es.add_relationship(cr)
        
    features, feature_names = ft.dfs(entityset=es, target_entity='PetID',
                                     max_depth=3,verbose=True)
    
    features = pd.merge(data[['PetID']], features.reset_index(), on='PetID', how='left')
    label_encode = LabelEncoder()
    for i in features.columns:
        if features[i].dtype =="object":
            features[i] = features[i].fillna('未知')
            features[i] = list(map(str, features[i]))
            features[i] = label_encode.fit_transform(features[i])
    features = features[['Breed1.NUM_UNIQUE(PetID.Vaccinated)',
     'FurLength.NUM_UNIQUE(PetID.Fee)',
     'Description.MODE(PetID.VideoAmt)',
     'Sterilized.NUM_UNIQUE(PetID.Fee)',
     'Description.NUM_UNIQUE(PetID.Gender)',
     'Description.NUM_UNIQUE(PetID.RescuerID)',
     'State.NUM_UNIQUE(PetID.Quantity)',
     'Color2.NUM_UNIQUE(PetID.RescuerID)',
     'Age.NUM_UNIQUE(PetID.Color3)',
     'Description.NUM_UNIQUE(PetID.MaturitySize)',
     'RescuerID.MODE(PetID.Health)',
     'Color3.NUM_UNIQUE(PetID.PhotoAmt)',
     'Color1.NUM_UNIQUE(PetID.Name)',
     'Breed1.MODE(PetID.Type)',
     'PhotoAmt.NUM_UNIQUE(PetID.Color1)',
     'MaturitySize.NUM_UNIQUE(PetID.Fee)',
     'Vaccinated.NUM_UNIQUE(PetID.RescuerID)',
     'State.MODE(PetID.Type)',
     'Color2.NUM_UNIQUE(PetID.Health)',
     'Color2.MODE(PetID.PhotoAmt)',
     'Dewormed.MODE(PetID.RescuerID)',
     'Quantity.NUM_UNIQUE(PetID.Description)',
     'Breed1.NUM_UNIQUE(PetID.Dewormed)',
     'Color1.NUM_UNIQUE(PetID.State)',
     'Color1.NUM_UNIQUE(PetID.VideoAmt)',
     'Fee.MODE(PetID.Age)',
     'Health.MODE(PetID.Description)',
     'Color3.NUM_UNIQUE(PetID.Health)',
     'Breed1.MODE(PetID.Breed2)',
     'Fee.NUM_UNIQUE(PetID.Color2)',
     'Color3.MODE(PetID.Type)',
     'Sterilized.NUM_UNIQUE(PetID.Name)',
     'Breed1.NUM_UNIQUE(PetID.FurLength)',
     'Breed2.NUM_UNIQUE(PetID.Sterilized)',
     'VideoAmt.MODE(PetID.PhotoAmt)',
     'Gender.NUM_UNIQUE(PetID.Name)',
     'Health.NUM_UNIQUE(PetID.Age)',
     'Age.NUM_UNIQUE(PetID.Color2)',
     'Description.NUM_UNIQUE(PetID.FurLength)',
     'Fee.NUM_UNIQUE(PetID.MaturitySize)',
     'Age.MODE(PetID.Color1)',
     'State.NUM_UNIQUE(PetID.RescuerID)',
     'Color3.NUM_UNIQUE(PetID.Fee)',
     'FurLength.NUM_UNIQUE(PetID.Name)',
     'Vaccinated.MODE(PetID.Dewormed)',
     'Color1.NUM_UNIQUE(PetID.PhotoAmt)',
     'Fee.NUM_UNIQUE(PetID.Vaccinated)',
     'Fee.NUM_UNIQUE(PetID.Color3)',
     'Breed1.NUM_UNIQUE(PetID.Type)',
     'State.NUM_UNIQUE(PetID.Color3)',
     'Quantity.NUM_UNIQUE(PetID.Fee)',
     'Health.NUM_UNIQUE(PetID.Breed1)',
     'Quantity.NUM_UNIQUE(PetID.Gender)',
     'Color3.MODE(PetID.Breed1)',
     'Age.NUM_UNIQUE(PetID.Color1)',
     'Color3.NUM_UNIQUE(PetID.RescuerID)',
     'Dewormed.NUM_UNIQUE(PetID.Fee)',
     'Quantity.MODE(PetID.Color2)',
     'Dewormed.MODE(PetID.Type)',
     'Color1.MODE(PetID.Gender)',
     'Gender.NUM_UNIQUE(PetID.RescuerID)',
     'Breed2.MODE(PetID.Color1)',
     'Breed2.NUM_UNIQUE(PetID.FurLength)',
     'VideoAmt.COUNT(PetID)',
     'State.NUM_UNIQUE(PetID.VideoAmt)',
     'MaturitySize.NUM_UNIQUE(PetID.Name)',
     'Age.MODE(PetID.FurLength)',
     'Sterilized.NUM_UNIQUE(PetID.PhotoAmt)',
     'Vaccinated.NUM_UNIQUE(PetID.VideoAmt)',
     'VideoAmt.MODE(PetID.RescuerID)',
     'Vaccinated.MODE(PetID.Name)',
     'Breed2.MODE(PetID.Gender)',
     'MaturitySize.MODE(PetID.Type)',
     'Description.NUM_UNIQUE(PetID.Sterilized)',
     'Type.COUNT(PetID)',
     'PhotoAmt.NUM_UNIQUE(PetID.Health)',
     'Sterilized.NUM_UNIQUE(PetID.Quantity)',
     'MaturitySize.MODE(PetID.PhotoAmt)',
     'Dewormed.NUM_UNIQUE(PetID.Name)',
     'Description.NUM_UNIQUE(PetID.Dewormed)',
     'Breed1.NUM_UNIQUE(PetID.Sterilized)',
     'Description.NUM_UNIQUE(PetID.Fee)',
     'Sterilized.MODE(PetID.Vaccinated)',
     'Fee.MODE(PetID.Sterilized)',
     'Breed2.MODE(PetID.Type)',
     'Quantity.NUM_UNIQUE(PetID.Name)',
     'Breed2.NUM_UNIQUE(PetID.Dewormed)',
     'Quantity.MODE(PetID.Age)',
     'Quantity.MODE(PetID.Dewormed)',
     'Color3.NUM_UNIQUE(PetID.Description)',
     'VideoAmt.NUM_UNIQUE(PetID.Age)',
     'Fee.NUM_UNIQUE(PetID.Color1)',
     'Gender.MODE(PetID.Name)',
     'VideoAmt.MODE(PetID.Vaccinated)',
     'Quantity.MODE(PetID.Breed1)',
     'Color1.MODE(PetID.Type)',
     'Fee.NUM_UNIQUE(PetID.Health)',
     'Quantity.NUM_UNIQUE(PetID.State)',
     'Breed1.MODE(PetID.Color3)',
     'Fee.MODE(PetID.FurLength)',
     'Breed2.NUM_UNIQUE(PetID.Type)',
     'Fee.NUM_UNIQUE(PetID.Sterilized)',
     'Health.NUM_UNIQUE(PetID.Breed2)',
     'Quantity.NUM_UNIQUE(PetID.Color1)',
     'State.MODE(PetID.Dewormed)',
     'Color1.NUM_UNIQUE(PetID.RescuerID)',
     'Description.NUM_UNIQUE(PetID.Health)',
     'Fee.MODE(PetID.Gender)',
     'Description.NUM_UNIQUE(PetID.State)',
     'Color3.MODE(PetID.Dewormed)',
     'VideoAmt.NUM_UNIQUE(PetID.Breed1)',
     'PhotoAmt.MODE(PetID.State)',
     'Type.NUM_UNIQUE(PetID.Age)',
     'Age.NUM_UNIQUE(PetID.Dewormed)',
     'State.NUM_UNIQUE(PetID.Health)',
     'Breed1.MODE(PetID.State)',
     'State.MODE(PetID.PhotoAmt)',
     'MaturitySize.NUM_UNIQUE(PetID.Quantity)',
     'FurLength.NUM_UNIQUE(PetID.Quantity)',
     'Breed1.NUM_UNIQUE(PetID.Gender)',
     'Sterilized.NUM_UNIQUE(PetID.RescuerID)',
     'VideoAmt.NUM_UNIQUE(PetID.Breed2)',
     'Description.NUM_UNIQUE(PetID.Type)',
     'Vaccinated.MODE(PetID.RescuerID)',
     'Health.MODE(PetID.Name)',
     'Color3.NUM_UNIQUE(PetID.VideoAmt)',
     'Fee.NUM_UNIQUE(PetID.FurLength)',
     'Color3.NUM_UNIQUE(PetID.Name)',
     'Health.NUM_UNIQUE(PetID.Fee)',
     'Dewormed.NUM_UNIQUE(PetID.Quantity)',
     'PhotoAmt.MODE(PetID.Breed1)',
     'FurLength.NUM_UNIQUE(PetID.RescuerID)',
     'Color3.NUM_UNIQUE(PetID.MaturitySize)',
     'Quantity.NUM_UNIQUE(PetID.PhotoAmt)',
     'Health.NUM_UNIQUE(PetID.Description)',
     'Age.NUM_UNIQUE(PetID.Type)',
     'Fee.MODE(PetID.Color1)',
     'Dewormed.NUM_UNIQUE(PetID.RescuerID)',
     'Type.NUM_UNIQUE(PetID.Breed2)',
     'Breed2.MODE(PetID.Dewormed)',
     'PhotoAmt.NUM_UNIQUE(PetID.Color2)',
     'Breed2.NUM_UNIQUE(PetID.Gender)',
     'Health.NUM_UNIQUE(PetID.Name)',
     'Vaccinated.MODE(PetID.Type)',
     'VideoAmt.NUM_UNIQUE(PetID.Description)',
     'PhotoAmt.NUM_UNIQUE(PetID.Dewormed)',
     'State.NUM_UNIQUE(PetID.Color1)',
     'Dewormed.NUM_UNIQUE(PetID.PhotoAmt)',
     'Age.NUM_UNIQUE(PetID.Sterilized)',
     'Fee.NUM_UNIQUE(PetID.Type)',
     'FurLength.NUM_UNIQUE(PetID.PhotoAmt)',
     'Type.NUM_UNIQUE(PetID.Fee)',
     'Age.NUM_UNIQUE(PetID.Vaccinated)',
     'Quantity.NUM_UNIQUE(PetID.Color2)',
     'MaturitySize.NUM_UNIQUE(PetID.RescuerID)',
     'Type.NUM_UNIQUE(PetID.Breed1)',
     'Sterilized.MODE(PetID.Name)',
     'Description.NUM_UNIQUE(PetID.VideoAmt)',
     'Age.NUM_UNIQUE(PetID.FurLength)',
     'PhotoAmt.MODE(PetID.Vaccinated)',
     'Quantity.NUM_UNIQUE(PetID.RescuerID)',
     'Quantity.NUM_UNIQUE(PetID.Sterilized)',
     'Vaccinated.NUM_UNIQUE(PetID.PhotoAmt)',
     'Age.NUM_UNIQUE(PetID.Gender)',
     'PhotoAmt.NUM_UNIQUE(PetID.FurLength)',
     'Breed2.MODE(PetID.Color3)',
     'FurLength.NUM_UNIQUE(PetID.State)',
     'Type.NUM_UNIQUE(PetID.Name)',
     'PhotoAmt.MODE(PetID.Age)',
     'Sterilized.NUM_UNIQUE(PetID.VideoAmt)',
     'Color2.MODE(PetID.Vaccinated)',
     'Type.NUM_UNIQUE(PetID.Description)',
     'Health.NUM_UNIQUE(PetID.VideoAmt)',
     'Dewormed.NUM_UNIQUE(PetID.VideoAmt)',
     'VideoAmt.NUM_UNIQUE(PetID.FurLength)',
     'FurLength.NUM_UNIQUE(PetID.VideoAmt)',
     'Health.NUM_UNIQUE(PetID.PhotoAmt)',
     'VideoAmt.NUM_UNIQUE(PetID.Color1)',
     'Breed2.MODE(PetID.State)',
     'MaturitySize.MODE(PetID.Gender)',
     'PhotoAmt.MODE(PetID.Gender)',
     'Color1.MODE(PetID.PhotoAmt)',
     'Fee.NUM_UNIQUE(PetID.Dewormed)',
     'MaturitySize.MODE(PetID.Vaccinated)',
     'Breed2.NUM_UNIQUE(PetID.Vaccinated)',
     'PhotoAmt.NUM_UNIQUE(PetID.Vaccinated)',
     'VideoAmt.MODE(PetID.Type)',
     'Dewormed.MODE(PetID.Vaccinated)',
     'FurLength.MODE(PetID.Breed1)',
     'Health.NUM_UNIQUE(PetID.Color3)',
     'Quantity.NUM_UNIQUE(PetID.Color3)',
     'Sterilized.NUM_UNIQUE(PetID.Vaccinated)',
     'Type.MODE(PetID.RescuerID)',
     'Sterilized.NUM_UNIQUE(PetID.Gender)',
     'Type.MODE(PetID.Age)',
     'Type.MODE(PetID.State)',
     'Type.MODE(PetID.Sterilized)',
     'Sterilized.NUM_UNIQUE(PetID.FurLength)',
     'Type.MODE(PetID.Breed2)',
     'Sterilized.NUM_UNIQUE(PetID.Health)',
     'Type.MODE(PetID.Breed1)',
     'Sterilized.NUM_UNIQUE(PetID.Type)',
     'Sterilized.NUM_UNIQUE(PetID.MaturitySize)',
     'Type.MODE(PetID.Color1)',
     'Type.MODE(PetID.Health)',
     'Type.MODE(PetID.MaturitySize)',
     'Type.MODE(PetID.Gender)',
     'Sterilized.NUM_UNIQUE(PetID.State)',
     'Type.MODE(PetID.FurLength)',
     'Type.MODE(PetID.Fee)',
     'Type.MODE(PetID.VideoAmt)',
     'Type.NUM_UNIQUE(PetID.Color1)',
     'Type.MODE(PetID.Name)',
     'Type.MODE(PetID.Dewormed)',
     'Type.MODE(PetID.PhotoAmt)',
     'Type.MODE(PetID.Description)',
     'Type.MODE(PetID.Color3)',
     'Type.MODE(PetID.Quantity)',
     'Type.MODE(PetID.Color2)',
     'Type.MODE(PetID.Vaccinated)',
     'Color1.MODE(PetID.Dewormed)',
     'Type.NUM_UNIQUE(PetID.Color2)',
     'Type.NUM_UNIQUE(PetID.Color3)',
     'VideoAmt.NUM_UNIQUE(PetID.PhotoAmt)',
     'VideoAmt.NUM_UNIQUE(PetID.Name)',
     'VideoAmt.NUM_UNIQUE(PetID.MaturitySize)',
     'VideoAmt.NUM_UNIQUE(PetID.Health)',
     'VideoAmt.NUM_UNIQUE(PetID.Gender)',
     'VideoAmt.NUM_UNIQUE(PetID.Fee)',
     'VideoAmt.NUM_UNIQUE(PetID.Dewormed)',
     'VideoAmt.NUM_UNIQUE(PetID.Color3)',
     'VideoAmt.MODE(PetID.Sterilized)',
     'VideoAmt.MODE(PetID.State)',
     'VideoAmt.MODE(PetID.Quantity)',
     'VideoAmt.MODE(PetID.MaturitySize)',
     'VideoAmt.MODE(PetID.Health)',
     'VideoAmt.MODE(PetID.Gender)',
     'VideoAmt.MODE(PetID.FurLength)',
     'VideoAmt.MODE(PetID.Fee)',
     'VideoAmt.MODE(PetID.Dewormed)',
     'VideoAmt.MODE(PetID.Color3)',
     'VideoAmt.MODE(PetID.Color2)',
     'VideoAmt.NUM_UNIQUE(PetID.Quantity)',
     'VideoAmt.NUM_UNIQUE(PetID.RescuerID)',
     'VideoAmt.NUM_UNIQUE(PetID.State)',
     'Breed1.MODE(PetID.Fee)',
     'Age.MODE(PetID.Breed2)',
     'Age.MODE(PetID.Color3)',
     'Age.MODE(PetID.Fee)',
     'Age.MODE(PetID.Health)',
     'Age.MODE(PetID.MaturitySize)',
     'Age.MODE(PetID.Quantity)',
     'Age.MODE(PetID.State)',
     'Age.MODE(PetID.VideoAmt)',
     'Breed1.MODE(PetID.Health)',
     'VideoAmt.NUM_UNIQUE(PetID.Sterilized)',
     'Breed1.MODE(PetID.Quantity)',
     'Breed1.MODE(PetID.VideoAmt)',
     'Breed2.MODE(PetID.Fee)',
     'Breed2.MODE(PetID.Health)',
     'Breed2.MODE(PetID.Quantity)',
     'Breed2.MODE(PetID.VideoAmt)',
     'VideoAmt.NUM_UNIQUE(PetID.Vaccinated)',
     'VideoAmt.NUM_UNIQUE(PetID.Type)',
     'VideoAmt.MODE(PetID.Color1)',
     'VideoAmt.MODE(PetID.Breed2)',
     'VideoAmt.MODE(PetID.Breed1)',
     'Type.NUM_UNIQUE(PetID.Sterilized)',
     'Vaccinated.MODE(PetID.Color3)',
     'Vaccinated.MODE(PetID.Color2)',
     'Vaccinated.MODE(PetID.Color1)',
     'Vaccinated.MODE(PetID.Breed2)',
     'Vaccinated.MODE(PetID.Breed1)',
     'Vaccinated.MODE(PetID.Age)',
     'Type.NUM_UNIQUE(PetID.VideoAmt)',
     'Type.NUM_UNIQUE(PetID.Vaccinated)',
     'Type.NUM_UNIQUE(PetID.State)',
     'Vaccinated.MODE(PetID.Fee)',
     'Type.NUM_UNIQUE(PetID.RescuerID)',
     'Type.NUM_UNIQUE(PetID.Quantity)',
     'Type.NUM_UNIQUE(PetID.PhotoAmt)',
     'Type.NUM_UNIQUE(PetID.MaturitySize)',
     'Type.NUM_UNIQUE(PetID.Health)',
     'Type.NUM_UNIQUE(PetID.Gender)',
     'Type.NUM_UNIQUE(PetID.FurLength)',
     'Type.NUM_UNIQUE(PetID.Dewormed)',
     'Vaccinated.MODE(PetID.Description)',
     'Vaccinated.MODE(PetID.FurLength)',
     'VideoAmt.MODE(PetID.Age)',
     'Vaccinated.NUM_UNIQUE(PetID.Color3)',
     'Vaccinated.NUM_UNIQUE(PetID.Type)',
     'Vaccinated.NUM_UNIQUE(PetID.Sterilized)',
     'Vaccinated.NUM_UNIQUE(PetID.State)',
     'Vaccinated.NUM_UNIQUE(PetID.MaturitySize)',
     'Vaccinated.NUM_UNIQUE(PetID.Health)',
     'Vaccinated.NUM_UNIQUE(PetID.Gender)',
     'Vaccinated.NUM_UNIQUE(PetID.FurLength)',
     'Vaccinated.NUM_UNIQUE(PetID.Dewormed)',
     'Vaccinated.NUM_UNIQUE(PetID.Color2)',
     'Vaccinated.MODE(PetID.Gender)',
     'Vaccinated.NUM_UNIQUE(PetID.Color1)',
     'Vaccinated.MODE(PetID.VideoAmt)',
     'Vaccinated.MODE(PetID.Sterilized)',
     'Vaccinated.MODE(PetID.State)',
     'Vaccinated.MODE(PetID.Quantity)',
     'Vaccinated.MODE(PetID.PhotoAmt)',
     'Vaccinated.MODE(PetID.MaturitySize)',
     'Vaccinated.MODE(PetID.Health)',
     'VideoAmt.NUM_UNIQUE(PetID.Color2)',
     'PhotoAmt.MODE(PetID.Breed2)',
     'Sterilized.NUM_UNIQUE(PetID.Dewormed)',
     'FurLength.MODE(PetID.Age)',
     'Fee.MODE(PetID.Health)',
     'Fee.MODE(PetID.MaturitySize)',
     'Fee.MODE(PetID.Quantity)',
     'Fee.MODE(PetID.State)',
     'Fee.MODE(PetID.VideoAmt)',
     'Fee.NUM_UNIQUE(PetID.Gender)',
     'FurLength.MODE(PetID.Breed2)',
     'FurLength.MODE(PetID.MaturitySize)',
     'FurLength.MODE(PetID.Color1)',
     'FurLength.MODE(PetID.Color2)',
     'FurLength.MODE(PetID.Color3)',
     'FurLength.MODE(PetID.Dewormed)',
     'FurLength.MODE(PetID.Fee)',
     'FurLength.MODE(PetID.Gender)',
     'Fee.MODE(PetID.Dewormed)',
     'Fee.MODE(PetID.Color3)',
     'Fee.MODE(PetID.Breed2)',
     'Dewormed.NUM_UNIQUE(PetID.Vaccinated)',
     'Dewormed.NUM_UNIQUE(PetID.Type)',
     'Dewormed.NUM_UNIQUE(PetID.Sterilized)',
     'Dewormed.NUM_UNIQUE(PetID.State)',
     'Dewormed.NUM_UNIQUE(PetID.MaturitySize)',
     'Dewormed.NUM_UNIQUE(PetID.Health)',
     'Dewormed.NUM_UNIQUE(PetID.Gender)',
     'Dewormed.NUM_UNIQUE(PetID.FurLength)',
     'Dewormed.NUM_UNIQUE(PetID.Color3)',
     'Dewormed.NUM_UNIQUE(PetID.Color2)',
     'Dewormed.NUM_UNIQUE(PetID.Color1)',
     'Dewormed.MODE(PetID.VideoAmt)',
     'Dewormed.MODE(PetID.Sterilized)',
     'Dewormed.MODE(PetID.State)',
     'FurLength.MODE(PetID.Health)',
     'FurLength.MODE(PetID.PhotoAmt)',
     'Dewormed.MODE(PetID.PhotoAmt)',
     'Gender.MODE(PetID.Fee)',
     'Gender.MODE(PetID.Breed2)',
     'Gender.MODE(PetID.Color1)',
     'Gender.MODE(PetID.Color2)',
     'Gender.MODE(PetID.Color3)',
     'Gender.MODE(PetID.Description)',
     'Gender.MODE(PetID.Dewormed)',
     'Gender.MODE(PetID.FurLength)',
     'FurLength.MODE(PetID.Quantity)',
     'Gender.MODE(PetID.Health)',
     'Gender.MODE(PetID.MaturitySize)',
     'Gender.MODE(PetID.PhotoAmt)',
     'Gender.MODE(PetID.Quantity)',
     'Gender.MODE(PetID.RescuerID)',
     'Gender.MODE(PetID.State)',
     'Gender.MODE(PetID.Breed1)',
     'Gender.MODE(PetID.Age)',
     'FurLength.NUM_UNIQUE(PetID.Vaccinated)',
     'FurLength.NUM_UNIQUE(PetID.Type)',
     'FurLength.NUM_UNIQUE(PetID.Sterilized)',
     'FurLength.NUM_UNIQUE(PetID.MaturitySize)',
     'FurLength.NUM_UNIQUE(PetID.Health)',
     'FurLength.NUM_UNIQUE(PetID.Gender)',
     'FurLength.NUM_UNIQUE(PetID.Dewormed)',
     'FurLength.NUM_UNIQUE(PetID.Color3)',
     'FurLength.NUM_UNIQUE(PetID.Color2)',
     'FurLength.NUM_UNIQUE(PetID.Color1)',
     'FurLength.MODE(PetID.VideoAmt)',
     'FurLength.MODE(PetID.Vaccinated)',
     'FurLength.MODE(PetID.Type)',
     'FurLength.MODE(PetID.Sterilized)',
     'FurLength.MODE(PetID.State)',
     'Dewormed.MODE(PetID.Quantity)',
     'Dewormed.MODE(PetID.Name)',
     'Sterilized.NUM_UNIQUE(PetID.Color3)',
     'Color2.MODE(PetID.MaturitySize)',
     'Color2.MODE(PetID.Color3)',
     'Color2.MODE(PetID.Dewormed)',
     'Color2.MODE(PetID.Fee)',
     'Color2.MODE(PetID.FurLength)',
     'Color2.MODE(PetID.Gender)',
     'Color2.MODE(PetID.Health)',
     'Color2.MODE(PetID.Quantity)',
     'Color2.NUM_UNIQUE(PetID.Sterilized)',
     'Color2.MODE(PetID.State)',
     'Color2.MODE(PetID.Sterilized)',
     'Color2.MODE(PetID.VideoAmt)',
     'Color2.NUM_UNIQUE(PetID.Dewormed)',
     'Color2.NUM_UNIQUE(PetID.FurLength)',
     'Color2.NUM_UNIQUE(PetID.Gender)',
     'Color2.MODE(PetID.Breed2)',
     'Color2.MODE(PetID.Age)',
     'Color1.NUM_UNIQUE(PetID.Vaccinated)',
     'Color1.NUM_UNIQUE(PetID.Type)',
     'Color1.NUM_UNIQUE(PetID.Sterilized)',
     'Color1.NUM_UNIQUE(PetID.MaturitySize)',
     'Color1.NUM_UNIQUE(PetID.Health)',
     'Color1.NUM_UNIQUE(PetID.Gender)',
     'Color1.NUM_UNIQUE(PetID.FurLength)',
     'Color1.NUM_UNIQUE(PetID.Dewormed)',
     'Color1.MODE(PetID.VideoAmt)',
     'Color1.MODE(PetID.Sterilized)',
     'Color1.MODE(PetID.State)',
     'Color1.MODE(PetID.Quantity)',
     'Color1.MODE(PetID.MaturitySize)',
     'Color1.MODE(PetID.Health)',
     'Color1.MODE(PetID.FurLength)',
     'Color2.NUM_UNIQUE(PetID.MaturitySize)',
     'Color2.NUM_UNIQUE(PetID.Type)',
     'Dewormed.MODE(PetID.MaturitySize)',
     'Dewormed.MODE(PetID.Breed2)',
     'Color3.NUM_UNIQUE(PetID.Sterilized)',
     'Color3.NUM_UNIQUE(PetID.Type)',
     'Color3.NUM_UNIQUE(PetID.Vaccinated)',
     'Color1.MODE(PetID.Color3)',
     'Dewormed.MODE(PetID.Age)',
     'Dewormed.MODE(PetID.Breed1)',
     'Dewormed.MODE(PetID.Color1)',
     'Color2.NUM_UNIQUE(PetID.Vaccinated)',
     'Dewormed.MODE(PetID.Color3)',
     'Dewormed.MODE(PetID.Description)',
     'Dewormed.MODE(PetID.Fee)',
     'Dewormed.MODE(PetID.FurLength)',
     'Dewormed.MODE(PetID.Gender)',
     'Dewormed.MODE(PetID.Health)',
     'Color3.NUM_UNIQUE(PetID.Gender)',
     'Color3.NUM_UNIQUE(PetID.FurLength)',
     'Color3.NUM_UNIQUE(PetID.Dewormed)',
     'Color3.MODE(PetID.VideoAmt)',
     'Color3.MODE(PetID.Vaccinated)',
     'Color3.MODE(PetID.Sterilized)',
     'Color3.MODE(PetID.State)',
     'Color3.MODE(PetID.Quantity)',
     'Color3.MODE(PetID.MaturitySize)',
     'Color3.MODE(PetID.Health)',
     'Color3.MODE(PetID.Gender)',
     'Color3.MODE(PetID.FurLength)',
     'Color3.MODE(PetID.Fee)',
     'Color3.MODE(PetID.Color2)',
     'Color3.MODE(PetID.Color1)',
     'Color3.MODE(PetID.Breed2)',
     'Color3.MODE(PetID.Age)',
     'Gender.MODE(PetID.Sterilized)',
     'Gender.MODE(PetID.Type)',
     'Gender.MODE(PetID.Vaccinated)',
     'Quantity.MODE(PetID.VideoAmt)',
     'Quantity.MODE(PetID.Gender)',
     'Quantity.MODE(PetID.Health)',
     'Quantity.MODE(PetID.MaturitySize)',
     'Quantity.MODE(PetID.State)',
     'Quantity.MODE(PetID.Sterilized)',
     'Quantity.MODE(PetID.Vaccinated)',
     'Quantity.NUM_UNIQUE(PetID.Dewormed)',
     'State.MODE(PetID.Color3)',
     'Quantity.NUM_UNIQUE(PetID.FurLength)',
     'Quantity.NUM_UNIQUE(PetID.Type)',
     'Quantity.NUM_UNIQUE(PetID.Vaccinated)',
     'RescuerID.NUM_UNIQUE(PetID.State)',
     'State.MODE(PetID.Breed2)',
     'State.MODE(PetID.Color1)',
     'Quantity.MODE(PetID.FurLength)',
     'Quantity.MODE(PetID.Fee)',
     'Quantity.MODE(PetID.Color1)',
     'Quantity.MODE(PetID.Breed2)',
     'PhotoAmt.NUM_UNIQUE(PetID.Type)',
     'PhotoAmt.NUM_UNIQUE(PetID.Sterilized)',
     'PhotoAmt.NUM_UNIQUE(PetID.Gender)',
     'PhotoAmt.MODE(PetID.VideoAmt)',
     'PhotoAmt.MODE(PetID.Sterilized)',
     'PhotoAmt.MODE(PetID.Quantity)',
     'PhotoAmt.MODE(PetID.MaturitySize)',
     'PhotoAmt.MODE(PetID.Health)',
     'PhotoAmt.MODE(PetID.FurLength)',
     'PhotoAmt.MODE(PetID.Fee)',
     'PhotoAmt.MODE(PetID.Dewormed)',
     'PhotoAmt.MODE(PetID.Color3)',
     'PhotoAmt.MODE(PetID.Color1)',
     'State.MODE(PetID.Color2)',
     'State.MODE(PetID.Fee)',
     'Gender.MODE(PetID.VideoAmt)',
     'Sterilized.MODE(PetID.MaturitySize)',
     'Sterilized.MODE(PetID.Color3)',
     'Sterilized.MODE(PetID.Dewormed)',
     'Sterilized.MODE(PetID.Fee)',
     'Sterilized.MODE(PetID.FurLength)',
     'Sterilized.MODE(PetID.Gender)',
     'Sterilized.MODE(PetID.Health)',
     'Sterilized.MODE(PetID.PhotoAmt)',
     'State.MODE(PetID.FurLength)',
     'Sterilized.MODE(PetID.Quantity)',
     'Sterilized.MODE(PetID.State)',
     'Sterilized.MODE(PetID.Type)',
     'Sterilized.MODE(PetID.VideoAmt)',
     'Sterilized.NUM_UNIQUE(PetID.Color1)',
     'Sterilized.NUM_UNIQUE(PetID.Color2)',
     'Sterilized.MODE(PetID.Color2)',
     'Sterilized.MODE(PetID.Color1)',
     'Sterilized.MODE(PetID.Breed2)',
     'Sterilized.MODE(PetID.Breed1)',
     'State.NUM_UNIQUE(PetID.Vaccinated)',
     'State.NUM_UNIQUE(PetID.Type)',
     'State.NUM_UNIQUE(PetID.Sterilized)',
     'State.NUM_UNIQUE(PetID.Gender)',
     'State.NUM_UNIQUE(PetID.FurLength)',
     'State.NUM_UNIQUE(PetID.Dewormed)',
     'State.NUM_UNIQUE(PetID.Color2)',
     'State.MODE(PetID.VideoAmt)',
     'State.MODE(PetID.Sterilized)',
     'State.MODE(PetID.Quantity)',
     'State.MODE(PetID.MaturitySize)',
     'State.MODE(PetID.Health)',
     'State.MODE(PetID.Gender)',
     'Color1.MODE(PetID.Fee)',
     'Color1.MODE(PetID.Age)',
     'Color1.MODE(PetID.Breed2)',
     'Health.MODE(PetID.Quantity)',
     'Health.MODE(PetID.Dewormed)',
     'Health.MODE(PetID.Fee)',
     'Health.MODE(PetID.FurLength)',
     'Health.MODE(PetID.Gender)',
     'Health.MODE(PetID.MaturitySize)',
     'Health.MODE(PetID.PhotoAmt)',
     'Health.MODE(PetID.RescuerID)',
     'MaturitySize.NUM_UNIQUE(PetID.Vaccinated)',
     'Health.MODE(PetID.State)',
     'Health.MODE(PetID.Sterilized)',
     'Health.MODE(PetID.Type)',
     'Health.MODE(PetID.Vaccinated)',
     'Health.MODE(PetID.VideoAmt)',
     'Health.NUM_UNIQUE(PetID.Color1)',
     'Health.MODE(PetID.Color3)',
     'Health.MODE(PetID.Color2)',
     'Health.MODE(PetID.Color1)',
     'Health.MODE(PetID.Breed2)',
     'Health.MODE(PetID.Breed1)',
     'Health.MODE(PetID.Age)',
     'Gender.NUM_UNIQUE(PetID.VideoAmt)',
     'Gender.NUM_UNIQUE(PetID.Vaccinated)',
     'Gender.NUM_UNIQUE(PetID.Type)',
     'Gender.NUM_UNIQUE(PetID.Sterilized)',
     'Gender.NUM_UNIQUE(PetID.MaturitySize)',
     'Gender.NUM_UNIQUE(PetID.Health)',
     'Gender.NUM_UNIQUE(PetID.FurLength)',
     'Gender.NUM_UNIQUE(PetID.Dewormed)',
     'Gender.NUM_UNIQUE(PetID.Color3)',
     'Gender.NUM_UNIQUE(PetID.Color2)',
     'Gender.NUM_UNIQUE(PetID.Color1)',
     'Health.NUM_UNIQUE(PetID.Color2)',
     'Health.NUM_UNIQUE(PetID.Dewormed)',
     'Health.NUM_UNIQUE(PetID.FurLength)',
     'MaturitySize.MODE(PetID.Health)',
     'MaturitySize.NUM_UNIQUE(PetID.Type)',
     'MaturitySize.NUM_UNIQUE(PetID.Sterilized)',
     'MaturitySize.NUM_UNIQUE(PetID.State)',
     'MaturitySize.NUM_UNIQUE(PetID.PhotoAmt)',
     'MaturitySize.NUM_UNIQUE(PetID.Health)',
     'MaturitySize.NUM_UNIQUE(PetID.Gender)',
     'MaturitySize.NUM_UNIQUE(PetID.FurLength)',
     'MaturitySize.NUM_UNIQUE(PetID.Dewormed)',
     'MaturitySize.NUM_UNIQUE(PetID.Color3)',
     'MaturitySize.NUM_UNIQUE(PetID.Color2)',
     'MaturitySize.NUM_UNIQUE(PetID.Color1)',
     'MaturitySize.MODE(PetID.Sterilized)',
     'MaturitySize.MODE(PetID.State)',
     'MaturitySize.MODE(PetID.Quantity)',
     'MaturitySize.MODE(PetID.FurLength)',
     'Health.NUM_UNIQUE(PetID.Gender)',
     'MaturitySize.MODE(PetID.Fee)',
     'MaturitySize.MODE(PetID.Dewormed)',
     'MaturitySize.MODE(PetID.Color3)',
     'MaturitySize.MODE(PetID.Color2)',
     'MaturitySize.MODE(PetID.Color1)',
     'MaturitySize.MODE(PetID.Breed2)',
     'MaturitySize.MODE(PetID.Age)',
     'Health.NUM_UNIQUE(PetID.Vaccinated)',
     'Health.NUM_UNIQUE(PetID.Type)',
     'Health.NUM_UNIQUE(PetID.Sterilized)',
     'Health.NUM_UNIQUE(PetID.State)',
     'Health.NUM_UNIQUE(PetID.RescuerID)',
     'Health.NUM_UNIQUE(PetID.Quantity)',
     'Health.NUM_UNIQUE(PetID.MaturitySize)',
     'MaturitySize.MODE(PetID.VideoAmt)']]
    
    new_columns = []
    for i in features.columns:
        new_columns.append('featuretools_' + i)
    features.columns = new_columns
    del data
    gc.collect()
    
    train_data_temp = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")
    test_data_temp = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")
    temp_data = pd.concat([train_data_temp, test_data_temp], axis=0)
    features['PetID'] = list(temp_data['PetID'])
    del train_data_temp,test_data_temp,temp_data
    gc.collect()
    return features
    

nurbs=get_feat3()
train = pd.merge(train1,nurbs, on='PetID', how='left')
test = pd.merge(test1, nurbs, on='PetID', how='left')
features = [x for x in train.columns if x not in ['Breed1',"breed","color","Breed2","State","lan_type","malai_type","Type","concat_text","is_group","Name",'PetID',"Description",'AdoptionSpeed']]

label='AdoptionSpeed'

params = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 80,
#          'max_depth':11,
          'learning_rate': 0.01,
          'bagging_fraction': 0.9,
           "bagging_freq":3,
          'feature_fraction': 0.4,
          'min_split_gain': 0.01,
#           'min_child_samples': 30,
#           "lambda_l1": 0.1,
          'verbosity': -1,
          'early_stop': 100,
          'verbose_eval': 200,
#           "random_state":1017,
          'num_rounds': 10000}


def runCAT(train_X, train_y, test_X, test_y, test_X2, params):
#     print('Prep LGB')
    d_train = Pool(train_X, label=train_y)
    d_valid = Pool(test_X, label=test_y)
    watchlist = (d_train, d_valid)
#     print('Train LGB')
    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    early_stop = None
    if params.get('early_stop'):
        early_stop = params.pop('early_stop')
    model = CatBoostRegressor(iterations=num_rounds, 
        learning_rate = 0.03,
        od_type='Iter',
         od_wait=early_stop,
        loss_function='RMSE',
        eval_metric='RMSE',
        bagging_temperature=0.9,                   
        random_seed = 2019,
        task_type='GPU'
                          )
    model.fit(d_train,eval_set=d_valid,
            use_best_model=True,
            verbose=verbose_eval
                         )
    
    print('Predict 1/2')
    pred_test_y = model.predict(test_X)
    log=0#log_loss(test_y,pred_test_y)
    print("log_loss:",log)
    class_list=[0,1,2,3,4]
#     pred_test_y=np.array([sum(pred_test_y[ix]*class_list) for
#                                ix in range(len(pred_test_y[:,0]))]) 
    optR = OptimizedRounder()
    optR.fit(pred_test_y, test_y)
    len_0 = sum([1 for i in test_y if i==0])
    coefficients = optR.coefficients()
    pred_test_y_k = optR.predict(pred_test_y, coefficients,len_0)
   
    print("Valid Counts = ", Counter(test_y))
    print("Predicted Counts = ", Counter(pred_test_y_k))
    print("Coefficients = ", coefficients)
    qwk = cohen_kappa_score(test_y, pred_test_y_k,weights='quadratic')
    print("QWK = ", qwk)
    print('Predict 2/2')
    pred_test_y2 = model.predict(test_X2)
#     pred_test_y2=np.array([sum(pred_test_y2[ix]*class_list) for
#                                ix in range(len(pred_test_y2[:,0]))]) 
   
    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), 0, coefficients, qwk,log
###model 5
results = run_cv_model(train[features], test[features], train[label], runLGB, params, rmse, 'LGB')
lgb5_train=[r[0] for r in results['train']]
lgb5_test=[r[0] for r in results['test']]
t5=time.time()
print("model5 cost:{} s".format(t5-t4))

train = pd.merge(train2, nurbs, on='PetID', how='left')
test = pd.merge(test2, nurbs, on='PetID', how='left')

del nurbs
gc.collect()
###model 6
features = [x for x in train.columns if x not in ['label_description',"is_group","Name",'PetID',"Description",'AdoptionSpeed']]
results = run_cv_model(train[features], test[features], train[label], runCAT, params, rmse, 'CAT')

cat2_train=[r[0] for r in results['train']]
cat2_test=[r[0] for r in results['test']]
t6=time.time()
print("model6 cost:{} s".format(t6-t5))


####model 7
features = [x for x in train.columns if x not in ["is_group","Name",'PetID',"Description",'AdoptionSpeed']]
params = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 80,
#          'max_depth':11,
          'learning_rate': 0.01,
          'bagging_fraction': 0.9,
           "bagging_freq":3,
          'feature_fraction': 0.4,
          'min_split_gain': 0.01,
#           'min_child_samples': 30,
#           "lambda_l1": 0.1,
          'verbosity': -1,
          'early_stop': 100,
          'verbose_eval': 200,
#           "random_state":1017,
          'num_rounds': 10000}
results = run_cv_model(train[features], test[features], train[label], runLGB, params, rmse, 'LGB')
lgb6_train=[r[0] for r in results['train']]
lgb6_test=[r[0] for r in results['test']]
del train,test,results
gc.collect()
t7=time.time()
print("model7 cost:{} s".format(t7-t6))

def nn1_model(train_img_feat=train_img_feat,test_img_feat=test_img_feat):
    train_data = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")
    test_data = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")
    
    breed=pd.read_csv("../input/petfinder-adoption-prediction/breed_labels.csv")
    color = pd.read_csv("../input/petfinder-adoption-prediction/color_labels.csv")
    
    color_dict = dict(zip(color['ColorID'].values.astype("str"),color['ColorName'].values))
    breed_dict = dict(zip(breed['BreedID'].values.astype("str"),breed['BreedName'].values))
    
    def get_purebreed_feat(df):
        if df['Breed2']==0 and df['Breed1']!=307:
            return 1
        return 0
    train_data['purebreed']=train_data.apply(lambda x:get_purebreed_feat(x),1)
    test_data['purebreed']=test_data.apply(lambda x:get_purebreed_feat(x),1)
    
    train_data['mixed']=train_data['Gender'].apply(lambda x:1 if x==3 else 0)
    test_data['mixed']=test_data['Gender'].apply(lambda x:1 if x==3 else 0)
    
    
    train_data['lan_type'] = train_data.Description.map(lambda x:lan_type(x))
    train_data['malai_type'] = train_data.Description.map(lambda x:malai_type(x))
    
    test_data['lan_type'] = test_data.Description.map(lambda x:lan_type(x))
    test_data['malai_type'] = test_data.Description.map(lambda x:malai_type(x))
    
    def name_deal(df):
        if "No Name" in df:
            return np.nan
        if df =="nan":
            return np.nan
        else:
            return df
    train_data['Name'] = train_data['Name'].apply(lambda x:name_deal(str(x)),1)
    test_data['Name'] = test_data['Name'].apply(lambda x:name_deal(str(x)),1)
    
    train_data['Name'],indexer=pd.factorize(train_data['Name'])
    test_data['Name'] = indexer.get_indexer(test_data['Name'])
    
    group = train_data['RescuerID'].values
    
    rescuer_df=train_data.groupby("RescuerID",as_index=False).count()[["RescuerID","PetID"]]
    rescuer_df.columns=["RescuerID","rescuer_cnt"]
    train_data=pd.merge(train_data,rescuer_df,on="RescuerID",how="left")
    train_data.drop("RescuerID",1,inplace=True)
    
    rescuer_df=test_data.groupby("RescuerID",as_index=False).count()[["RescuerID","PetID"]]
    rescuer_df.columns=["RescuerID","rescuer_cnt"]
    test_data=pd.merge(test_data,rescuer_df,on="RescuerID",how="left")
    test_data.drop("RescuerID",1,inplace=True)
    
    train_data["num_chars"] = train_data["Description"].apply(lambda x: len(str(x)))
    test_data["num_chars"] = test_data["Description"].apply(lambda x: len(str(x)))
    
    train_data['Description'] = train_data['Description'].fillna("None")
    test_data['Description'] = test_data['Description'].fillna("None")
    
    train_data["Description"] = train_data["Description"].str.lower()
    test_data["Description"] = test_data["Description"].str.lower()
    
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
     '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
     '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
     '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
     '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
    def clean_text(x):
        for punct in puncts:
            x = x.replace(punct, f' {punct} ')
        return x
    
    
    train_data["Description"] = train_data["Description"].apply(lambda x: clean_text(x))
    test_data["Description"] = test_data["Description"].apply(lambda x: clean_text(x))
    

    def deal_desc(df):
        if df['lan_type']==1:
            return "null"
        if df['lan_type']==3:
            text=jieba.cut(df['Description'])
            text=" ".join(text)
            text=text.replace("   "," ")
            return text
        else:
            return df['Description']
    train_data['Description']=train_data.apply(lambda x:deal_desc(x), 1)
    test_data['Description']=test_data.apply(lambda x:deal_desc(x), 1)
    
    doc_sent_mag = []
    doc_sent_score = []
    nf_count = 0
    for pet in train_data['PetID'].values:
        try:
            with open('../input/petfinder-adoption-prediction/train_sentiment/' + pet + '.json', 'r') as f:
                sentiment = json.load(f)
            doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
            doc_sent_score.append(sentiment['documentSentiment']['score'])
        except FileNotFoundError:
            nf_count += 1
            doc_sent_mag.append(-1)
            doc_sent_score.append(-1)
    
    train_data['doc_sent_mag'] = doc_sent_mag
    train_data['doc_sent_score'] = doc_sent_score
    
    doc_sent_mag = []
    doc_sent_score = []
    nf_count = 0
    for pet in test_data['PetID'].values:
        try:
            with open('../input/petfinder-adoption-prediction/test_sentiment/' + pet + '.json', 'r') as f:
                sentiment = json.load(f)
            doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
            doc_sent_score.append(sentiment['documentSentiment']['score'])
        except FileNotFoundError:
            nf_count += 1
            doc_sent_mag.append(-1)
            doc_sent_score.append(-1)
    
    test_data['doc_sent_mag'] = doc_sent_mag
    test_data['doc_sent_score'] = doc_sent_score
    
    vertex_xs = []
    vertex_ys = []
    bounding_confidences = []
    bounding_importance_fracs = []
    dominant_blues = []
    dominant_greens = []
    dominant_reds = []
    dominant_pixel_fracs = []
    dominant_scores = []
    label_descriptions = []
    label_scores = []
    nf_count = 0
    nl_count = 0
    for pet in train_data['PetID'].values:
        try:
            with open('../input/petfinder-adoption-prediction/train_metadata/' + pet + '-1.json', 'r') as f:
                data = json.load(f)
            vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
            vertex_xs.append(vertex_x)
            vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
            vertex_ys.append(vertex_y)
            bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
            bounding_confidences.append(bounding_confidence)
            bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
            bounding_importance_fracs.append(bounding_importance_frac)
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('blue',-1)
            dominant_blues.append(dominant_blue)
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('green',-1)
            dominant_greens.append(dominant_green)
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('red',-1)
            dominant_reds.append(dominant_red)
            dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
            dominant_pixel_fracs.append(dominant_pixel_frac)
            dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
            dominant_scores.append(dominant_score)
            if data.get('labelAnnotations'):
                label_description = data['labelAnnotations'][0]['description']
                label_descriptions.append(label_description)
                label_score = data['labelAnnotations'][0]['score']
                label_scores.append(label_score)
            else:
                nl_count += 1
                label_descriptions.append('nothing')
                label_scores.append(-1)
        except FileNotFoundError:
            nf_count += 1
            vertex_xs.append(-1)
            vertex_ys.append(-1)
            bounding_confidences.append(-1)
            bounding_importance_fracs.append(-1)
            dominant_blues.append(-1)
            dominant_greens.append(-1)
            dominant_reds.append(-1)
            dominant_pixel_fracs.append(-1)
            dominant_scores.append(-1)
            label_descriptions.append('nothing')
            label_scores.append(-1)
    
    print(nf_count)
    print(nl_count)
    train_data[ 'vertex_x'] = vertex_xs
    train_data['vertex_y'] = vertex_ys
    train_data['bounding_confidence'] = bounding_confidences
    train_data['bounding_importance'] = bounding_importance_fracs
    train_data['dominant_blue'] = dominant_blues
    train_data['dominant_green'] = dominant_greens
    train_data['dominant_red'] = dominant_reds
    train_data['dominant_pixel_frac'] = dominant_pixel_fracs
    train_data['dominant_score'] = dominant_scores
    train_data['label_description'] = label_descriptions
    train_data['label_score'] = label_scores
    
    
    vertex_xs = []
    vertex_ys = []
    bounding_confidences = []
    bounding_importance_fracs = []
    dominant_blues = []
    dominant_greens = []
    dominant_reds = []
    dominant_pixel_fracs = []
    dominant_scores = []
    label_descriptions = []
    label_scores = []
    nf_count = 0
    nl_count = 0
    for pet in test_data['PetID'].values:
        try:
            with open('../input/petfinder-adoption-prediction/test_metadata/' + pet + '-1.json', 'r') as f:
                data = json.load(f)
            vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
            vertex_xs.append(vertex_x)
            vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
            vertex_ys.append(vertex_y)
            bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
            bounding_confidences.append(bounding_confidence)
            bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
            bounding_importance_fracs.append(bounding_importance_frac)
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('blue',-1)
            dominant_blues.append(dominant_blue)
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('green',-1)
            dominant_greens.append(dominant_green)
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('red',-1)
            dominant_reds.append(dominant_red)
            dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
            dominant_pixel_fracs.append(dominant_pixel_frac)
            dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
            dominant_scores.append(dominant_score)
            if data.get('labelAnnotations'):
                label_description = data['labelAnnotations'][0]['description']
                label_descriptions.append(label_description)
                label_score = data['labelAnnotations'][0]['score']
                label_scores.append(label_score)
            else:
                nl_count += 1
                label_descriptions.append('nothing')
                label_scores.append(-1)
        except FileNotFoundError:
            nf_count += 1
            vertex_xs.append(-1)
            vertex_ys.append(-1)
            bounding_confidences.append(-1)
            bounding_importance_fracs.append(-1)
            dominant_blues.append(-1)
            dominant_greens.append(-1)
            dominant_reds.append(-1)
            dominant_pixel_fracs.append(-1)
            dominant_scores.append(-1)
            label_descriptions.append('nothing')
            label_scores.append(-1)
    
    print(nf_count)
    test_data[ 'vertex_x'] = vertex_xs
    test_data['vertex_y'] = vertex_ys
    test_data['bounding_confidence'] = bounding_confidences
    test_data['bounding_importance'] = bounding_importance_fracs
    test_data['dominant_blue'] = dominant_blues
    test_data['dominant_green'] = dominant_greens
    test_data['dominant_red'] = dominant_reds
    test_data['dominant_pixel_frac'] = dominant_pixel_fracs
    test_data['dominant_score'] = dominant_scores
    test_data['label_description'] = label_descriptions
    test_data['label_score'] = label_scores
    
    del  vertex_xs,vertex_ys,bounding_confidences,bounding_importance_fracs,dominant_blues,dominant_greens,dominant_reds,dominant_pixel_fracs,dominant_scores
    del label_descriptions,label_scores,doc_sent_mag,doc_sent_score
    gc.collect()
    
    train_data['label_description'] =train_data['label_description'].astype(np.str)
    test_data['label_description'] =test_data['label_description'].astype(np.str)
    
    def get_text(df):
        x=""
        if df['Type']==1:
            x+="dog"+" "
        if df['Type']==2:
            x+="cat"+" "
        for i in ['Breed1',"Breed2"]:
            if df[i]==0:
                continue
            x+=breed_dict[str(df[i])]+" "
        for i in ["Color1","Color2","Color3"]:
            if df[i]==0:
                continue
            x+=color_dict[str(df[i])]+" "
        x+=df['label_description']+" "
        x=x+df['Description']
        return x
    train_data['Description']=train_data.apply(lambda x:get_text(x),1)
    test_data['Description']=test_data.apply(lambda x:get_text(x),1)
    
    text_list = train_data['Description'].values.tolist()
    text_list.extend(test_data['Description'].values.tolist())
    
    documents = text_list
    texts = [[word for word in str(document).split(' ') ] for document in documents]
    
    
    
    w2v = Word2Vec(texts, size=128, window=7, iter=8, seed=10, workers=2, min_count=3)
    w2v.wv.save_word2vec_format('w2v_128.txt')
    print("w2v model done")
    del w2v
    gc.collect()
    embed_size = 128 # how big is each word vector
    max_features = None # how many unique words to use (i.e num rows in embedding vector)
    maxlen = 230 # max number of words in a question to use
    
    ## Tokenize the sentences
    train_X = train_data["Description"].values
    test_X = test_data["Description"].values
    
    tokenizer = Tokenizer(num_words=max_features, filters='')
    tokenizer.fit_on_texts(list(train_X)+list(test_X))
    
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)
    
    ## Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)
    
    ## Get the target values
    train_y = train_data['AdoptionSpeed'].values
    
    word_index=tokenizer.word_index
    features = [x for x in train_data.columns if x not in ["num_words","num_unique_words","num_stopwords","num_punctuations","mean_word_len",'label_description',"Name", 'PetID', "Description", 'AdoptionSpeed']]
    
    cate_col=['Breed1', 'Breed2', 'Color1', 'Color2', 'Color3', 'State']
    onehot_col=['Type','Gender','MaturitySize','MaturitySize','FurLength','Vaccinated',
               'Dewormed','Sterilized','Health','purebreed','Color1', 'Color2', 'Color3', ]
    num_col=features
    
   
    
    sc = StandardScaler()
    data = pd.concat([train_data, test_data])
    sc.fit(data[num_col])
    del data
    gc.collect()
    train_data[num_col] = sc.transform(train_data[num_col])
    test_data[num_col] = sc.transform(test_data[num_col])
    train_num_feat = train_data[num_col]
    test_num_feat = test_data[num_col]
    
    train_img_feat.reset_index(inplace=True)
    test_img_feat.reset_index(inplace=True)
    train_img_feat.columns = ["PetID"]+["img_"+str(i) for i in range(train_img_feat.shape[1]-1)]
    test_img_feat.columns = ["PetID"]+["img_"+str(i) for i in range(train_img_feat.shape[1]-1)]
    del train_img_feat['PetID'], test_img_feat['PetID']
    train_num_feat = pd.concat([train_num_feat, train_img_feat], axis=1).values
    test_num_feat = pd.concat([test_num_feat, test_img_feat], axis=1).values
    
    embedding_matrix=get_embedding_matrix(word_index)
    
    def hybrid_model(embedding_matrix):
        K.clear_session()
        inp_text = Input(shape=(maxlen, ))
        emb = Embedding(
            input_dim=embedding_matrix.shape[0],
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix],
            input_length=maxlen,
            trainable=False)(inp_text)
        x = SpatialDropout1D(rate=0.22)(emb)
        x = Bidirectional(CuDNNLSTM(120, return_sequences=True, kernel_initializer=glorot_uniform(seed=123)))(x)  
        x1 = Conv1D(filters=100, kernel_size=1, kernel_initializer=glorot_uniform(seed=123),
                       padding='same', activation='relu')(x)
        x2 = Conv1D(filters=90, kernel_size=2, kernel_initializer=glorot_uniform(seed=123),
                       padding='same', activation='relu')(x)
        x3 = Conv1D(filters=30, kernel_size=3, kernel_initializer=glorot_uniform(seed=123),
                       padding='same', activation='relu')(x)
        x4 = Conv1D(filters=10, kernel_size=5, kernel_initializer=glorot_uniform(seed=123),
                       padding='same', activation='relu')(x)
    
        x1 = GlobalMaxPool1D()(x1)
        x2 = GlobalMaxPool1D()(x2)
        x3 = GlobalMaxPool1D()(x3)
        x4 = GlobalMaxPool1D()(x4)
        x5 = AttentionWeightedAverage()(x)
        
        inp_num = Input(shape=(293, ))
        x = concatenate([x1, x2, x3, x4, x5, inp_num])
        x = Dense(200, kernel_initializer='glorot_uniform', activation=gelu)(x)
        #x = PReLU()(x)
        x = Dropout(0.22)(x)
        x = BatchNormalization()(x)
        x = Dense(200, kernel_initializer='glorot_uniform', activation=gelu)(x)
        #x = PReLU()(x)
        x = Dropout(0.22)(x)
        x = BatchNormalization()(x)
        out = Dense(1, kernel_initializer=glorot_uniform(seed=123))(x)
    
        model = Model(inputs=[inp_text, inp_num], outputs=out)
        model.compile(loss='mean_squared_error', optimizer=AdamW(weight_decay=0.02))
        return model
    
    kfold = StratifiedKFold(n_splits=5, random_state=1017, shuffle=True)
    pred_oof=np.zeros((train_X.shape[0], ))
    y_test = np.zeros((test_X.shape[0], ))
    cv_scores = []
    qwk_scores = []
    all_coefficients = np.zeros((5, 4))
    
    for i, (train_index, test_index) in enumerate(kfold.split(train_X, train_y)):
        print("FOLD | {}/{}".format(i+1,5))
        X_tr, X_vl, X_tr2, X_vl2, y_tr, y_vl = train_X[train_index], train_X[test_index], train_num_feat[
            train_index], train_num_feat[test_index], train_y[train_index], train_y[test_index]
        #X_tr0 = get_keras_data(X_trall,  cate_col)
        #X_tr0['text']=X_tr
        #X_tr0['num']=X_tr2
        #X_vl0 = get_keras_data(X_vlall,  cate_col)
        #X_vl0['text']=X_vl
        #X_vl0['num']=X_vl2
        filepath="weights_best.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.00001, verbose=2)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=2, mode='auto')
        callbacks = [checkpoint, reduce_lr, earlystopping]
        model = hybrid_model(embedding_matrix)
        if i == 0:print(model.summary()) 
        model.fit([X_tr, X_tr2], y_tr, batch_size=128, epochs=20, validation_data=([X_vl, X_vl2], y_vl), verbose=2, callbacks=callbacks,)
        model.load_weights(filepath)    
        y_pred = np.squeeze(model.predict([X_vl, X_vl2], batch_size=256, verbose=2))
        pred_oof[test_index] = y_pred
        y_test += np.squeeze(model.predict([test_X, test_num_feat], batch_size=256, verbose=2))/5
        optR = OptimizedRounder()
        optR.fit(y_pred, y_vl)
        len_0 = sum([1 for i in y_vl if i==0])
        coefficients = optR.coefficients()
        pred_test_y_k = optR.predict(y_pred, coefficients, len_0)
        print("Valid Counts = ", Counter(y_vl))
        print("Predicted Counts = ", Counter(pred_test_y_k))
        print("Coefficients = ", coefficients)
        
        qwk = cohen_kappa_score(y_vl, pred_test_y_k,weights='quadratic')
        cv_score = rmse(y_vl, y_pred)
        cv_scores.append(cv_score)
        qwk_scores.append(qwk)
        all_coefficients[i, :] = coefficients
        print( ' cv score {}: RMSE {} QWK {}'.format(i+1, cv_score, qwk))
        print("##"*40)
        
    print('cv mean RMSE score : {}'.format( np.mean(cv_scores)))
    print('cv std RMSE score : {}'.format( np.std(cv_scores)))
    print('cv mean QWK score : {}'.format( np.mean(qwk_scores)))
    print('cv std QWK score : {}'.format( np.std(qwk_scores)))  
    
    del train_num_feat,test_num_feat,train_X,test_X
    gc.collect()
    
    nn1_train = [r for r in pred_oof]
    nn1_test = [r for r in y_test]
    
    return nn1_train,nn1_test,embedding_matrix,train_img_feat,test_img_feat,train_data,test_data

###model 8
###nn1
nn1_train,nn1_test,embedding_matrix,train_img_feat,test_img_feat,train_data,test_data=nn1_model()
t8=time.time()
print("model8 cost:{} s".format(t8-t7))
####model 9
###nn2
def nn2_model(train,test,embedding_matrix,train_img_feat,test_img_feat):
    
    embed_size = 128 # how big is each word vector
    max_features = None # how many unique words to use (i.e num rows in embedding vector)
    maxlen = 220 # max number of words in a question to use
    
    ## Tokenize the sentences
    train_X = train["concat_text"].values
    test_X = test["concat_text"].values
    
    tokenizer = Tokenizer(num_words=max_features, filters='')
    tokenizer.fit_on_texts(list(train_X)+list(test_X))
    
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)
    
    ## Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)
    
    ## Get the target values
    train_y = train['AdoptionSpeed'].values
    
    word_index = tokenizer.word_index
    features = [x for x in train.columns if x not in ["num_words","num_unique_words","num_stopwords","num_punctuations","mean_word_len",'Breed1',"breed","color","Breed2","State","concat_text",'label_description',"Name",'PetID',"Description",'AdoptionSpeed']]

    num_col = features
    
    sc = StandardScaler()
    data = pd.concat([train, test])
    sc.fit(data[num_col])
    del data
    gc.collect()
    train[num_col] = sc.transform(train[num_col])
    test[num_col] = sc.transform(test[num_col])
    train_num_feat = train[num_col]
    test_num_feat = test[num_col]
    
    train_num_feat = pd.concat([train_num_feat, train_img_feat], axis=1).values
    test_num_feat = pd.concat([test_num_feat, test_img_feat], axis=1).values
    
    
    def hybrid_model(embedding_matrix=embedding_matrix, sp=0.22, filters=[96, 100, 30], weight_decay=0.01):
        K.clear_session()
        inp_text = Input(shape=(maxlen, ))
        emb = Embedding(
            input_dim=embedding_matrix.shape[0],
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix],
            input_length=maxlen,
            trainable=False)(inp_text)
        x = SpatialDropout1D(rate=sp, seed=1024)(emb)
        x = Bidirectional(CuDNNLSTM(128, return_sequences=True, kernel_initializer=glorot_uniform(seed=123), 
                                    recurrent_initializer=orthogonal(gain=1.0, seed=10000)))(x)
        #xx = Bidirectional(CuDNNGRU(60, return_sequences=False, kernel_initializer=glorot_uniform(seed=123)))(x)
        #x1 = Conv1D(filters=filters[0], kernel_size=1, kernel_initializer=glorot_uniform(seed=123),
        #               padding='same', activation='relu')(x)
        c = Conv1D(filters=filters[1], kernel_size=2, kernel_initializer=glorot_uniform(seed=123),
                       padding='same', activation='relu')(x)
        #x3 = Conv1D(filters=filters[2], kernel_size=3, kernel_initializer=glorot_uniform(seed=123),
        #               padding='same', activation='relu')(x)
        #x4 = Conv1D(filters=10, kernel_size=5, kernel_initializer=glorot_uniform(seed=123),
        #               padding='same', activation='relu')(x)
    
        #x1 = GlobalMaxPool1D()(x1)
        x2 = GlobalMaxPool1D()(c)
        x3 = GlobalAvgPool1D()(c)
        #x3 = GlobalMaxPool1D()(x3)
        #x4 = GlobalMaxPool1D()(x4)
        x5 = AttentionWeightedAverage()(x)
        
        inp_num = Input(shape=(test_num_feat.shape[1], ))
        x = concatenate([x2, x3, x5, inp_num])
        x = Dense(200, kernel_initializer=glorot_uniform(seed=123), activation=gelu
                 )(x)
        #x = PReLU()(x)
        x = Dropout(0.23, seed=1024)(x)
        #x = BatchNormalization()(x)
        #x = Dense(200, kernel_initializer=glorot_uniform(seed=123), activation=gelu)(x)
        #x = PReLU()(x)
        #x = Dropout(0.23, seed=1024)(x)
        #x = BatchNormalization()(x)
        out = Dense(1, kernel_initializer=glorot_uniform(seed=123))(x)
    
        model = Model(inputs=[inp_text, inp_num], outputs=out)
        model.compile(loss='mean_squared_error', optimizer=AdamW(weight_decay=weight_decay))
        #model.compile(loss='mean_squared_error', optimizer='rmsprop')
        return model
    kfold = StratifiedKFold(n_splits=5, random_state=1017, shuffle=True)
    pred_oof=np.zeros((train_X.shape[0], ))
    y_test = np.zeros((test_X.shape[0], ))
    cv_scores = []
    qwk_scores = []
    all_coefficients = np.zeros((5, 4))
    
    for i, (train_index, test_index) in enumerate(kfold.split(train_X, train_y)):
        print("FOLD | {}/{}".format(i+1,5))
        X_tr, X_vl, X_tr2, X_vl2, y_tr, y_vl = train_X[train_index], train_X[test_index], train_num_feat[
            train_index], train_num_feat[test_index], train_y[train_index], train_y[test_index]
        
        filepath="weights_best.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.00001, verbose=2)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=2, mode='auto')
        callbacks = [checkpoint, reduce_lr, earlystopping]
        if i == 0:
            model = hybrid_model(embedding_matrix=embedding_matrix, sp=0.22, filters=[96, 100, 30], weight_decay=0.04)
        elif i == 1:
            model = hybrid_model(embedding_matrix=embedding_matrix, sp=0.22, filters=[96, 100, 30], weight_decay=0.04)
        elif i == 2:
            model = hybrid_model(embedding_matrix=embedding_matrix, sp=0.22, filters=[96, 100, 30], weight_decay=0.04)
        elif i == 3:
            model = hybrid_model(embedding_matrix=embedding_matrix, sp=0.22, filters=[96, 100, 30], weight_decay=0.04)
        elif i == 4:
            model = hybrid_model(embedding_matrix=embedding_matrix, sp=0.22, filters=[96, 100, 30], weight_decay=0.04)
        if i == 0:print(model.summary()) 
        model.fit([X_tr, X_tr2], y_tr, batch_size=128, epochs=20, validation_data=([X_vl, X_vl2], y_vl), verbose=2, callbacks=callbacks,)
        model.load_weights(filepath)    
        y_pred = np.squeeze(model.predict([X_vl, X_vl2], batch_size=256, verbose=2))
        pred_oof[test_index] = y_pred
        y_test += np.squeeze(model.predict([test_X, test_num_feat], batch_size=256, verbose=2))/5
        optR = OptimizedRounder()
        optR.fit(y_pred, y_vl)
        len_0 = sum([1 for i in y_vl if i==0])
        coefficients = optR.coefficients()
        pred_test_y_k = optR.predict(y_pred, coefficients, len_0)
        print("Valid Counts = ", Counter(y_vl))
        print("Predicted Counts = ", Counter(pred_test_y_k))
        print("Coefficients = ", coefficients)
        qwk = cohen_kappa_score(y_vl, pred_test_y_k,weights='quadratic')
        cv_score = rmse(y_vl, y_pred)
        cv_scores.append(cv_score)
        qwk_scores.append(qwk)
        all_coefficients[i, :] = coefficients
        print( ' cv score {}: RMSE {} QWK {}'.format(i+1, cv_score, qwk))
        print("##"*40)
        
    print('cv mean RMSE score : {}'.format( np.mean(cv_scores)))
    print('cv std RMSE score : {}'.format( np.std(cv_scores)))
    print('cv mean QWK score : {}'.format( np.mean(qwk_scores)))
    print('cv std QWK score : {}'.format( np.std(qwk_scores)))
    
    nn2_train = [r for r in pred_oof]
    nn2_test = [r for r in y_test]
    del train_X,test_X
    gc.collect()
    
    return nn2_train,nn2_test,train_num_feat,test_num_feat
nn2_train,nn2_test,train_num_feat,test_num_feat=nn2_model(train1,test1,embedding_matrix,train_img_feat,test_img_feat)
del train_img_feat,test_img_feat
gc.collect()
t9=time.time()
print("model9 cost:{} s".format(t9-t8))
####model 10
###nn3

def nn3_model(train,test,embedding_matrix,train_num_feat,test_num_feat):
    maxlen = 200
    max_features = None 
    train_X = train["concat_text"].values
    test_X = test["concat_text"].values
    
    tokenizer = Tokenizer(num_words=max_features, filters='')
    tokenizer.fit_on_texts(list(train_X)+list(test_X))
    
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)
    
    ## Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)
    
     ## Get the target values
    train_y = train['AdoptionSpeed'].values
    
    def hybrid_model(embedding_matrix):
        K.clear_session()
        inp_text = Input(shape=(maxlen, ))
        emb = Embedding(
            input_dim=embedding_matrix.shape[0],
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix],
            input_length=maxlen,
            trainable=False)(inp_text)
        x = SpatialDropout1D(rate=0.22)(emb)
        x = Bidirectional(CuDNNLSTM(120, return_sequences=True, kernel_initializer=glorot_uniform(seed=123)))(x)  
        x1 = Conv1D(filters=96, kernel_size=1, kernel_initializer=glorot_uniform(seed=123),
                       padding='same', activation='relu')(x)
        x2 = Conv1D(filters=90, kernel_size=2, kernel_initializer=glorot_uniform(seed=123),
                       padding='same', activation='relu')(x)
        x3 = Conv1D(filters=30, kernel_size=3, kernel_initializer=glorot_uniform(seed=123),
                       padding='same', activation='relu')(x)
        x4 = Conv1D(filters=10, kernel_size=5, kernel_initializer=glorot_uniform(seed=123),
                       padding='same', activation='relu')(x)
    
        x1 = GlobalMaxPool1D()(x1)
        x2 = GlobalMaxPool1D()(x2)
        x3 = GlobalMaxPool1D()(x3)
        x4 = GlobalMaxPool1D()(x4)
        x5 = AttentionWeightedAverage()(x)
        
        inp_num = Input(shape=(test_num_feat.shape[1], ))
        x = concatenate([x1, x2, x3, x5, inp_num])
        x = Dense(200, kernel_initializer='glorot_uniform', activation=gelu)(x)
        #x = PReLU()(x)
        x = Dropout(0.22)(x)
        x = BatchNormalization()(x)
        x = Dense(200, kernel_initializer='glorot_uniform', activation=gelu)(x)
        #x = PReLU()(x)
        x = Dropout(0.22)(x)
        x = BatchNormalization()(x)
        out = Dense(5, activation="softmax",kernel_initializer=glorot_uniform(seed=123))(x)
    
        model = Model(inputs=[inp_text, inp_num], outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer=AdamW(weight_decay=0.02))
        return model
    
   
    kfold = StratifiedKFold(n_splits=5, random_state=1017, shuffle=True)
    pred_oof=np.zeros((train_X.shape[0], ))
    y_test = np.zeros((test_X.shape[0],))
    cv_scores = []
    qwk_scores = []
    all_coefficients = np.zeros((5, 4))
    
    y_label= to_categorical(train['AdoptionSpeed'])
    for i, (train_index, test_index) in enumerate(kfold.split(train_X, train_y)):
        print("FOLD | {}/{}".format(i+1,5))
        X_tr, X_vl, X_tr2, X_vl2, y_tr, y_vl = train_X[train_index], train_X[test_index], train_num_feat[
            train_index], train_num_feat[test_index], y_label[train_index], y_label[test_index]
        
        filepath="weights_best.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.00001, verbose=2)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=2, mode='auto')
        callbacks = [checkpoint, reduce_lr, earlystopping]
        model = hybrid_model(embedding_matrix)
        if i == 0:print(model.summary()) 
        model.fit([X_tr, X_tr2], y_tr, batch_size=128, epochs=20, validation_data=([X_vl, X_vl2], y_vl), verbose=2, callbacks=callbacks,)
        model.load_weights(filepath)  
        class_list=[0,1,2,3,4]
        y_pred = np.squeeze(model.predict([X_vl, X_vl2], batch_size=256, verbose=2))
        y_pred=np.array([sum(y_pred[ix]*class_list) for
                                   ix in range(len(y_pred[:,0]))]) 
        pred_oof[test_index] = y_pred
        test_temp = np.squeeze(model.predict([test_X, test_num_feat], batch_size=256, verbose=2))
    
        
        test_temp=np.array([sum(test_temp[ix]*class_list) for
                                   ix in range(len(test_temp[:,0]))]) 
        y_test+=np.squeeze(test_temp)/5
        y_vl= train_y[test_index]
        optR = OptimizedRounder()
        optR.fit(y_pred, y_vl)
        len_0 = sum([1 for i in y_vl if i==0])
        coefficients = optR.coefficients()
        pred_test_y_k = optR.predict(y_pred, coefficients, len_0)
        print("Valid Counts = ", Counter(y_vl))
        print("Predicted Counts = ", Counter(pred_test_y_k))
        print("Coefficients = ", coefficients)
        qwk = cohen_kappa_score(y_vl, pred_test_y_k,weights='quadratic')
        cv_score = rmse(y_vl, y_pred)
        cv_scores.append(cv_score)
        qwk_scores.append(qwk)
        all_coefficients[i, :] = coefficients
        print( ' cv score {}: RMSE {} QWK {}'.format(i+1, cv_score, qwk))
        print("##"*40)
        
    print('cv mean RMSE score : {}'.format( np.mean(cv_scores)))
    print('cv std RMSE score : {}'.format( np.std(cv_scores)))
    print('cv mean QWK score : {}'.format( np.mean(qwk_scores)))
    print('cv std QWK score : {}'.format( np.std(qwk_scores)))
    
    nn3_train = [r for r in pred_oof]
    nn3_test = [r for r in y_test]
    
    del train_X,test_X
    gc.collect()
    return  nn3_train,nn3_test 
nn3_train,nn3_test=nn3_model(train1,test1,embedding_matrix,train_num_feat,test_num_feat)
del embedding_matrix,train_num_feat,test_num_feat
gc.collect()
t10=time.time()
print("model10 cost:{} s".format(t10-t9))

######weak model###############################################
data = pd.concat([train_data,test_data])
data.index=range(len(data))
data_id=data['PetID'].values

del train_desc,test_desc
gc.collect()  

cols = [x for x in train1.columns if x not in ['Breed1',"breed","color","Breed2","State",'label_description',"lan_type","malai_type","Type","concat_text","is_group","Name",'PetID',"Description",'AdoptionSpeed']]

train1[cols]=train1[cols].fillna(0)
test1[cols]=test1[cols].fillna(0)
############################ 切分数据集 ##########################
print('开始进行一些前期处理')
train_feature = train1[cols].values
test_feature = test1[cols].values
    # 五则交叉验证
n_folds = 5
print('处理完毕')
df_stack3 = pd.DataFrame()
df_stack3['PetID']=data['PetID']
for label in ["AdoptionSpeed"]:
    score = train_data[label]
    
   
    ########################### SGD(随机梯度下降) ################################
    # print('sgd stacking')
    # stack_train = np.zeros((len(train_data),1))
    # stack_test = np.zeros((len(test_data),1))
    # score_va = 0

    # sk = StratifiedKFold( n_splits=5, random_state=1017,shuffle=True)
    # for i, (tr, va) in enumerate(sk.split(train_feature, score)):
    #     print('stack:%d/%d' % ((i + 1), n_folds))
    #     sgd = SGDRegressor(random_state=1017,)
    #     sgd.fit(train_feature[tr], score[tr])
    #     score_va = sgd.predict(train_feature[va])
    #     score_te = sgd.predict(test_feature)
    #     print('得分' + str(mean_squared_error(score[va], sgd.predict(train_feature[va]))))
    #     stack_train[va,0] = score_va
    #     stack_test[:,0]+= score_te
    # stack_test /= n_folds
    # stack = np.vstack([stack_train, stack_test])
#     df_stack3['tfidf_sgd_classfiy_{}'.format("feat1")] = stack[:,0]


    ########################### pac(PassiveAggressiveClassifier) ################################
    # print('PAC stacking')
    # stack_train = np.zeros((len(train_data),1))
    # stack_test = np.zeros((len(test_data),1))
    # score_va = 0

    # sk = StratifiedKFold( n_splits=5, random_state=1017,shuffle=True)
    # for i, (tr, va) in enumerate(sk.split(train_feature, score)):
    #     print('stack:%d/%d' % ((i + 1), n_folds))
    #     pac = PassiveAggressiveRegressor(random_state=1017)
    #     pac.fit(train_feature[tr], score[tr])
    #     score_va = pac.predict(train_feature[va])
    #     score_te = pac.predict(test_feature)
      
    #     print('得分' + str(mean_squared_error(score[va], pac.predict(train_feature[va]))))
    #     stack_train[va,0] = score_va
    #     stack_test[:,0] += score_te
    # stack_test /= n_folds
    # stack = np.vstack([stack_train, stack_test])

#     df_stack3['tfidf_pac_classfiy_{}'.format("feat1")] = stack[:,0]
    


    

    ########################### FTRL ################################
    print('MultinomialNB stacking')
    stack_train = np.zeros((len(train_data),1))
    stack_test = np.zeros((len(test_data),1))
    score_va = 0

    sk = StratifiedKFold( n_splits=5, random_state=1017,shuffle=True)
    for i, (tr, va) in enumerate(sk.split(train_feature, score)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        clf = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=train_feature.shape[1], iters=50, inv_link="identity", threads=1)
        clf.fit(train_feature[tr], score[tr])
        score_va = clf.predict(train_feature[va])
        score_te = clf.predict(test_feature)
      
        print('得分' + str(mean_squared_error(score[va], clf.predict(train_feature[va]))))
        stack_train[va,0] = score_va
        stack_test[:,0] += score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    
    df_stack3['tfidf_FTRL_classfiy_{}'.format("feat1")] = stack[:,0]
    
    ########################### ridge(RidgeClassfiy) ################################
    print('RidgeClassfiy stacking')
    stack_train = np.zeros((len(train_data),1))
    stack_test = np.zeros((len(test_data),1))
    score_va = 0

    sk = StratifiedKFold( n_splits=5, random_state=1017,shuffle=True)
    for i, (tr, va) in enumerate(sk.split(train_feature, score)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        ridge = Ridge(solver="sag", fit_intercept=True, random_state=42, alpha=30) 
        ridge.fit(train_feature[tr], score[tr])
        score_va = ridge.predict(train_feature[va])
        score_te = ridge.predict(test_feature)
       
        print('得分' + str(mean_squared_error(score[va], ridge.predict(train_feature[va]))))
        stack_train[va,0] = score_va
        stack_test[:,0] += score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])

    df_stack3['tfidf_ridge_classfiy_{}'.format("feat1")] = stack[:,0]
    
    ############################ Linersvc(LinerSVC) ################################
    # print('LinerSVC stacking')
    # stack_train = np.zeros((len(train_data),1))
    # stack_test = np.zeros((len(test_data),1))
    # score_va = 0

    # sk = StratifiedKFold( n_splits=5, random_state=1017,shuffle=True)
    # for i, (tr, va) in enumerate(sk.split(train_feature, score)):
    #     print('stack:%d/%d' % ((i + 1), n_folds))
    #     lsvc = LinearSVR(random_state=1017)
    #     lsvc.fit(train_feature[tr], score[tr])
    #     score_va = lsvc.predict(train_feature[va])
    #     score_te = lsvc.predict(test_feature)
       
    #     print('得分' + str(mean_squared_error(score[va], lsvc.predict(train_feature[va]))))
    #     stack_train[va,0] = score_va
    #     stack_test[:,0] += score_te
    # stack_test /= n_folds
    # stack = np.vstack([stack_train, stack_test])

#     df_stack3['tfidf_lsvc_classfiy_{}'.format("feat1")] = stack[:,0]
del stack,stack_train, stack_test,train_feature,test_feature
gc.collect()   
# df_stack.to_csv('graph_tfidf_classfiy.csv', index=None, encoding='utf8')
print('tfidf特征已保存\n')
del train1,test1
gc.collect()

cols = [x for x in train2.columns if x not in ['label_description',"is_group","Name",'PetID',"Description",'AdoptionSpeed']]
train2[cols]=train2[cols].fillna(0)
test2[cols]=test2[cols].fillna(0)
############################ 切分数据集 ##########################
print('开始进行一些前期处理')
train_feature = train2[cols].values
test_feature = test2[cols].values
    # 五则交叉验证
n_folds = 5
print('处理完毕')
df_stack4 = pd.DataFrame()
df_stack4['PetID']=data['PetID']
for label in ["AdoptionSpeed"]:
    score = train_data[label]
    
   
    ########################### SGD(随机梯度下降) ################################
    # print('sgd stacking')
    # stack_train = np.zeros((len(train_data),1))
    # stack_test = np.zeros((len(test_data),1))
    # score_va = 0

    # sk = StratifiedKFold( n_splits=5, random_state=1017,shuffle=True)
    # for i, (tr, va) in enumerate(sk.split(train_feature, score)):
    #     print('stack:%d/%d' % ((i + 1), n_folds))
    #     sgd = SGDRegressor(random_state=1017,)
    #     sgd.fit(train_feature[tr], score[tr])
    #     score_va = sgd.predict(train_feature[va])
    #     score_te = sgd.predict(test_feature)
    #     print('得分' + str(mean_squared_error(score[va], sgd.predict(train_feature[va]))))
    #     stack_train[va,0] = score_va
    #     stack_test[:,0]+= score_te
    # stack_test /= n_folds
    # stack = np.vstack([stack_train, stack_test])
#     df_stack4['tfidf_sgd_classfiy_{}'.format("feat2")] = stack[:,0]


    ########################### pac(PassiveAggressiveClassifier) ################################
    # print('PAC stacking')
    # stack_train = np.zeros((len(train_data),1))
    # stack_test = np.zeros((len(test_data),1))
    # score_va = 0

    # sk = StratifiedKFold( n_splits=5, random_state=1017,shuffle=True)
    # for i, (tr, va) in enumerate(sk.split(train_feature, score)):
    #     print('stack:%d/%d' % ((i + 1), n_folds))
    #     pac = PassiveAggressiveRegressor(random_state=1017)
    #     pac.fit(train_feature[tr], score[tr])
    #     score_va = pac.predict(train_feature[va])
    #     score_te = pac.predict(test_feature)
      
    #     print('得分' + str(mean_squared_error(score[va], pac.predict(train_feature[va]))))
    #     stack_train[va,0] = score_va
    #     stack_test[:,0] += score_te
    # stack_test /= n_folds
    # stack = np.vstack([stack_train, stack_test])

#     df_stack4['tfidf_pac_classfiy_{}'.format("feat2")] = stack[:,0]
    


    

    ########################### FTRL ################################
    print('MultinomialNB stacking')
    stack_train = np.zeros((len(train_data),1))
    stack_test = np.zeros((len(test_data),1))
    score_va = 0

    sk = StratifiedKFold( n_splits=5, random_state=1017,shuffle=True)
    for i, (tr, va) in enumerate(sk.split(train_feature, score)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        clf = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=train_feature.shape[1], iters=50, inv_link="identity", threads=1)
        clf.fit(train_feature[tr], score[tr])
        score_va = clf.predict(train_feature[va])
        score_te = clf.predict(test_feature)
      
        print('得分' + str(mean_squared_error(score[va], clf.predict(train_feature[va]))))
        stack_train[va,0] = score_va
        stack_test[:,0] += score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    
    df_stack4['tfidf_FTRL_classfiy_{}'.format("feat2")] = stack[:,0]
    
    ########################### ridge(RidgeClassfiy) ################################
    print('RidgeClassfiy stacking')
    stack_train = np.zeros((len(train_data),1))
    stack_test = np.zeros((len(test_data),1))
    score_va = 0

    sk = StratifiedKFold( n_splits=5, random_state=1017,shuffle=True)
    for i, (tr, va) in enumerate(sk.split(train_feature, score)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        ridge = Ridge(solver="sag", fit_intercept=True, random_state=42, alpha=30) 
        ridge.fit(train_feature[tr], score[tr])
        score_va = ridge.predict(train_feature[va])
        score_te = ridge.predict(test_feature)
       
        print('得分' + str(mean_squared_error(score[va], ridge.predict(train_feature[va]))))
        stack_train[va,0] = score_va
        stack_test[:,0] += score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])

    df_stack4['tfidf_ridge_classfiy_{}'.format("feat2")] = stack[:,0]
    
    ############################ Linersvc(LinerSVC) ################################
    # print('LinerSVC stacking')
    # stack_train = np.zeros((len(train_data),1))
    # stack_test = np.zeros((len(test_data),1))
    # score_va = 0

    # sk = StratifiedKFold( n_splits=5, random_state=1017,shuffle=True)
    # for i, (tr, va) in enumerate(sk.split(train_feature, score)):
    #     print('stack:%d/%d' % ((i + 1), n_folds))
    #     lsvc = LinearSVR(random_state=1017)
    #     lsvc.fit(train_feature[tr], score[tr])
    #     score_va = lsvc.predict(train_feature[va])
    #     score_te = lsvc.predict(test_feature)
       
    #     print('得分' + str(mean_squared_error(score[va], lsvc.predict(train_feature[va]))))
    #     stack_train[va,0] = score_va
    #     stack_test[:,0] += score_te
    # stack_test /= n_folds
    # stack = np.vstack([stack_train, stack_test])

#     df_stack4['tfidf_lsvc_classfiy_{}'.format("feat2")] = stack[:,0]
del stack,stack_train, stack_test,train_feature,test_feature
gc.collect()   
# df_stack.to_csv('graph_tfidf_classfiy.csv', index=None, encoding='utf8')
print('tfidf特征已保存\n')

# wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
#                                                               "hash_ngrams_weights": [1.5, 1.0],
#                                                               "hash_size": 2 ** 29,
#                                                               "norm": None,
#                                                               "tf": 'binary',
#                                                               "idf": None,
#                                                               }), procs=8)
# x_train = wb.fit_transform(train2['Description'])
# x_test = wb.transform(test2['Description'])

################
# Remove features with document frequency <=100
#@eg:1)
#mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 100, 0, 1), dtype=bool)
#sparse_merge = sparse_merge[:, mask]
#@eg:2)
#mask
#mask = np.where(X_param1_train.getnnz(axis=0) > 3)[0]
#X_param1_train = X_param1_train[:, mask]
################
# mask = np.array(np.clip(x_train.getnnz(axis=0) - 3, 0, 1), dtype=bool)
# x_train=x_train[:,mask]
# x_test=x_test[:,mask]

# sk = StratifiedKFold( n_splits=5, random_state=1017,shuffle=True)
# stack_train = np.zeros((len(train_data)))
# stack_test = np.zeros((len(test_data)))
# print("FTRL...")
# n_fold=5
# for i, (tr, va) in enumerate(sk.split(x_train, train_data['AdoptionSpeed'])):
#     print("FOLD | {}/{}".format(i+1,n_fold))
#     clf=  FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=x_train.shape[1], iters=50, inv_link="identity", threads=1)
#     clf.fit(x_train[tr],train_data['AdoptionSpeed'][tr])
#     score_va = clf.predict(x_train[va])
#     score_te = clf.predict(x_test)
#     stack_train[va] = score_va
#     stack_test += score_te
# stack_test /= n_fold

# train_data['FTRL_pred']=stack_train
# test_data['FTRL_pred']=stack_test
# print("Ridge...")
# stack_train = np.zeros((len(train_data)))
# stack_test = np.zeros((len(test_data)))
# for i, (tr, va) in enumerate(sk.split(x_train, train_data['AdoptionSpeed'])):
#     print("FOLD | {}/{}".format(i+1,n_fold))
#     clf= Ridge(solver="sag", fit_intercept=True, random_state=42, alpha=30)
#     clf.fit(x_train[tr],train_data['AdoptionSpeed'][tr])
#     score_va = clf.predict(x_train[va])
#     score_te = clf.predict(x_test)
#     stack_train[va] = score_va
#     stack_test += score_te
# stack_test /= n_fold

# train_data['ridge_pred']=stack_train
# test_data['ridge_pred']=stack_test

# print("LinearSVR...")
# stack_train = np.zeros((len(train_data)))
# stack_test = np.zeros((len(test_data)))
# for i, (tr, va) in enumerate(sk.split(x_train, train_data['AdoptionSpeed'])):
#     print("FOLD | {}/{}".format(i+1,n_fold))
#     clf= LinearSVR(random_state=1017)
#     clf.fit(x_train[tr],train_data['AdoptionSpeed'][tr])
#     score_va = clf.predict(x_train[va])
#     score_te = clf.predict(x_test)
#     stack_train[va] = score_va
#     stack_test += score_te
# stack_test /= n_fold

# train_data['svr_pred']=stack_train
# test_data['svr_pred']=stack_test

# print("pac...")
# stack_train = np.zeros((len(train_data)))
# stack_test = np.zeros((len(test_data)))
# for i, (tr, va) in enumerate(sk.split(x_train, train_data['AdoptionSpeed'])):
#     print("FOLD | {}/{}".format(i+1,n_fold))
#     clf= PassiveAggressiveRegressor(random_state=1017)
#     clf.fit(x_train[tr],train_data['AdoptionSpeed'][tr])
#     score_va = clf.predict(x_train[va])
#     score_te = clf.predict(x_test)
#     stack_train[va] = score_va
#     stack_test += score_te
# stack_test /= n_fold

# train_data['pac_pred']=stack_train
# test_data['pac_pred']=stack_test

# print("sgd...")
# stack_train = np.zeros((len(train_data)))
# stack_test = np.zeros((len(test_data)))
# for i, (tr, va) in enumerate(sk.split(x_train, train_data['AdoptionSpeed'])):
#     print("FOLD | {}/{}".format(i+1,n_fold))
#     clf= SGDRegressor(random_state=1017,)
#     clf.fit(x_train[tr],train_data['AdoptionSpeed'][tr])
#     score_va = clf.predict(x_train[va])
#     score_te = clf.predict(x_test)
#     stack_train[va] = score_va
#     stack_test += score_te
# stack_test /= n_fold

# train_data['sgd_pred']=stack_train
# test_data['sgd_pred']=stack_test

# del x_train,x_test,stack_train,stack_test
del train2,test2
gc.collect()

##3sigma 
def feat4_model():

    def read_sent_json(pet_id,data_source='train'):
    
        doc_sent_mag = []
        doc_sent_score = []
        nf_count = 0
        for pet in pet_id:
            try:
                with open(data_source + '_sentiment/' + pet + '.json', 'r') as f:
                    sentiment = json.load(f)
                doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
                doc_sent_score.append(sentiment['documentSentiment']['score'])
            except FileNotFoundError:
                nf_count += 1
                doc_sent_mag.append(-np.nan)
                doc_sent_score.append(-np.nan)
        return doc_sent_mag,doc_sent_score
    
    def read_meta_json(data_source='train'):
    
        vertex_xs = []
        vertex_ys = []
        bounding_confidences = []
        bounding_importance_fracs = []
        dominant_blues = []
        dominant_greens = []
        dominant_reds = []
        dominant_pixel_fracs = []
        dominant_scores = []
        label_descriptions = []
        label_all_descriptions = []
        label_scores = []
        file_name = []
        nl_count = 0
        file_list = os.listdir(data_source+'_metadata/')
        for file_i in file_list:
            file_name.append(file_i) 
            with open(data_source + '_metadata/' + file_i, 'r') as f:
                data = json.load(f)
            vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
            vertex_xs.append(vertex_x)
            vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
            vertex_ys.append(vertex_y)
            bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
            bounding_confidences.append(bounding_confidence)
            bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
            bounding_importance_fracs.append(bounding_importance_frac)
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('blue',-1)
            dominant_blues.append(dominant_blue)
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('green',-1)
            dominant_greens.append(dominant_green)
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('red',-1)
            dominant_reds.append(dominant_red)
            dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
            dominant_pixel_fracs.append(dominant_pixel_frac)
            dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
            dominant_scores.append(dominant_score)
            if data.get('labelAnnotations'):
                label_description = data['labelAnnotations'][0]['description']
                label_all_description = ' '.join([i['description'].replace(' ','_') for i in data['labelAnnotations']])
                label_descriptions.append(label_description)
                label_all_descriptions.append(label_all_description)
                label_score = data['labelAnnotations'][0]['score']
                label_scores.append(label_score)
            else:
                nl_count += 1
                label_all_descriptions.append('nothing')
                label_descriptions.append('nothing')
                label_scores.append(-1)
        out_df = pd.DataFrame({'file_name':file_name,
                               'vertex_xs':vertex_xs,
                               'vertex_ys':vertex_ys,
                               'bounding_confidences':bounding_confidences,
                               'bounding_importance_fracs':bounding_importance_fracs,
                               'dominant_blues':dominant_blues,
                               'dominant_greens':dominant_greens,
                               'dominant_reds':dominant_reds,
                               'dominant_pixel_fracs':dominant_pixel_fracs,
                               'dominant_scores':dominant_scores,
                               'label_descriptions':label_descriptions,
                               'label_all_descriptions':label_all_descriptions,
                               'label_scores':label_scores
                              })
        out_df['PetID'] = out_df['file_name'].str.split('-').str[0]
        out_df['PicID'] =  out_df['file_name'].str.split('[-|.]').str[1]
        
        return out_df
    
    breed = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')
    state = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')
    color = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')
    
    data_tr = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
    data_te = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
    
    def deal_breed(df):
        if df['Breed1']==df['Breed2']:
            df['Breed2']=0
        if df['Breed1']!=307 & df['Breed2']==307:
            temp=df["Breed1"]
            df['Breed1']=df['Breed2']
            df['Breed2']=temp
        return df
    data_tr=data_tr.apply(lambda x:deal_breed(x),1)
    data_te=data_te.apply(lambda x:deal_breed(x),1)
    
    breed_dict = dict(zip(breed['BreedID'].values.astype("str"),breed['BreedName'].values))
    def get_breed(df):
        
        x=""
        for i in ["Breed1","Breed2"]:
            if df[i]==0:
                continue
            x+=breed_dict[str(df[i])]+" "
        return x
    data_tr['breed']=data_tr.apply(lambda x:get_breed(x),1)
    data_te['breed']=data_te.apply(lambda x:get_breed(x),1)
    
    data_tr['Breed1_str']=data_tr['Breed1'].astype("str")
    data_tr['Breed1_str']=data_tr['Breed1'].map(breed_dict)
    data_tr['Breed1_str']=data_tr['Breed1'].replace(np.nan,"null")
    
    data_te['Breed1_str']=data_te['Breed1'].astype("str")
    data_te['Breed1_str']=data_te['Breed1'].map(breed_dict)
    data_te['Breed1_str']=data_te['Breed1'].replace(np.nan,"null")
    
    mean_encoder = MeanEncoder( categorical_features=['Breed1_str', 'breed'],target_type ='regression')
    
    data_tr = mean_encoder.fit_transform(data_tr, data_tr['AdoptionSpeed'])
    data_te = mean_encoder.transform(data_te)
    
    breed_encode_tr = data_tr[['Breed1_str_pred','breed_pred']]
    breed_encode_te = data_te[['Breed1_str_pred','breed_pred']]
    
    data_tr['doc_sent_mag'],data_tr['doc_sent_score'] = read_sent_json(data_tr['PetID'],data_source='../input/petfinder-adoption-prediction/train')
    data_te['doc_sent_mag'],data_te['doc_sent_score'] = read_sent_json(data_te['PetID'],data_source='../input/petfinder-adoption-prediction/test')
    
    meta_tr = read_meta_json(data_source='../input/petfinder-adoption-prediction/train')
    meta_te = read_meta_json(data_source='../input/petfinder-adoption-prediction/test')
    
    num_cols_0 = ['Age','MaturitySize','FurLength','Health','Quantity','Fee','VideoAmt','PhotoAmt']
    sent_cols_0 = ['doc_sent_mag','doc_sent_score']
    meta_cols_0 = ['vertex_xs', 'vertex_ys', 'bounding_confidences',
           'bounding_importance_fracs', 'dominant_blues', 'dominant_greens',
           'dominant_reds', 'dominant_pixel_fracs', 'dominant_scores','label_scores'] #'label_descriptions'
    cat_cols_0 = ['Type','Gender','Vaccinated','Dewormed','Sterilized']
    join_cols_0 = ['Breed1','Breed2','Color1','Color2','Color3','State']
    
    n_tr = data_tr.shape[0]
    n_te = data_te.shape[0]
    
    def data_process(data,meta_data,data_source='train'):
        fea_data = data[['PetID']+num_cols_0+sent_cols_0+cat_cols_0].copy()
        
        meta_1 = meta_data.query('PicID =="1"')
        temp = data[['PetID']].merge(meta_1,how='left',on='PetID')[meta_cols_0].fillna(-1)
        fea_data = pd.concat([fea_data,temp],axis=1)[['PetID']+num_cols_0+sent_cols_0+meta_cols_0+cat_cols_0]
        
    #     temp = meta_1.pivot(index='PetID',columns='label_descriptions',values='label_scores')
    #     temp = meta_data.loc[meta_data['PicID'].isin(['1']),:].groupby(['PetID','label_descriptions'])['label_scores'].max().reset_index()
    #     temp = temp.pivot(index='PetID',columns='label_descriptions',values='label_scores')
    #     temp.columns = [i.replace(' ','_') + '_scores' for i in temp.columns]
    #     fea_data = fea_data.merge(temp,how='left',on='PetID')
        
        
        for i in ['Age','MaturitySize','FurLength','Health',
                  'Fee','VideoAmt','PhotoAmt',
                  'Breed2','Color2','Color3']:
            fea_data[i+'_is_0'] = data[i] == 0
        for i in ['Fee','VideoAmt','PhotoAmt']:
            fea_data[i+'_avg'] = data[i] / data['Quantity']
        
        dummy_fea = pd.get_dummies(data[cat_cols_0],columns=cat_cols_0)
        fea_data = pd.concat([fea_data,dummy_fea],axis=1)
        fea_data['color_cnt'] = (data[['Color1','Color2','Color3']] != 0 ).sum(axis=1)
        fea_data['no_name'] = data['Name'].isnull() | data['Name'].str.contains('No Name')
        
        # Description
        temp = data['Description']
        fea_data['desc_ch'] = temp.str.count(u'[\u4e00-\u9fa5]')
        fea_data['desc_count'] = temp.str.count(u'[\u4e00-\u9fa5]') + temp.replace(u'[\u4e00-\u9fa5]','').str.count(' ')
        fea_data['desc_ch_rate'] = fea_data['desc_ch'] / fea_data['desc_count']
        
        # RescuerID
        fea_data['rescuer_cnt'] = data.groupby('RescuerID')['PetID'].transform('count')
        
        temp = data.groupby(['RescuerID','Type'])['PetID'].count().rename('rescuer_type_cnt').reset_index()
        temp = temp.pivot(index='RescuerID',columns='Type',values='rescuer_type_cnt').fillna(0)
        temp.columns = ['rescuer_type1_cnt','rescuer_type2_cnt']
        temp = data[['RescuerID']].merge(temp,how='left',on='RescuerID')
        fea_data[['rescuer_type1_cnt','rescuer_type2_cnt']] = temp[['rescuer_type1_cnt','rescuer_type2_cnt']] 
        fea_data['rescuer_type1_rate']= fea_data['rescuer_type1_cnt'] / fea_data['rescuer_cnt']
        
    # #     temp = data.groupby(['RescuerID','State'])['PetID'].count() / data_tr.groupby(['State'])['PetID'].count()
    # #     temp = temp.rename('Rescuer_State_rate').reset_index().drop('State',axis=1)
    # #     fea_data['Rescuer_State_rate']= data_tr[['RescuerID']].merge(temp,how='left',on='RescuerID')['Rescuer_State_rate']
        
        # Breed
        temp = data_tr.groupby(['Breed1'])['PetID'].count().rename('Breed1_cnt').reset_index()
        temp = data[['Breed1']].merge(temp,how='left',on='Breed1')   
        fea_data['Breed1_cnt'] = temp['Breed1_cnt']
        
        
        data_tr['Fee_avg'] = data_tr['Fee'] / data_tr['Quantity']
        temp = data_tr.groupby(['Breed1'])['Fee_avg'].mean().rename('Breed1_Fee_avg').reset_index()
        temp = data[['Breed1']].merge(temp,how='left',on='Breed1') 
        fea_data['Breed1_Fee_avg'] = temp['Breed1_Fee_avg']
        fea_data['Breed1_Fee_avg_diff'] = fea_data['Fee_avg'] - fea_data['Breed1_Fee_avg']
        
    
        
        
        
        
        # State
        
        temp = data_tr.groupby(['State','Type'])['PetID'].count().rename('state_type_cnt').reset_index()
        temp = temp.pivot(index='State',columns='Type',values='state_type_cnt').fillna(0)
        # temp.columns = ['state_type1_cnt','state_type2_cnt']
        temp.columns = ['state_type1_cnt_rank','state_type2_cnt_rank']
        temp['state_type1_cnt_rank'] = temp['state_type1_cnt_rank'].rank()
        temp['state_type2_cnt_rank'] = temp['state_type2_cnt_rank'].rank()
        temp = data[['State']].merge(temp,how='left',on='State')
        fea_data[['state_type1_cnt_rank','state_type2_cnt_rank']] = temp[['state_type1_cnt_rank','state_type2_cnt_rank']] 
        
        
        
        if data_source == 'train':
            label = data['AdoptionSpeed'].values
        else:
            label = None
        return fea_data,label
    
    fea_tr,label_tr = data_process(data_tr,meta_tr,data_source='train')
    fea_te,_        = data_process(data_te,meta_te,data_source='test')
    
    base_fea_cols = [i for i in fea_te.columns if i in fea_tr.columns]
    fea_tr = fea_tr[base_fea_cols]
    fea_te = fea_te[base_fea_cols]
    
    print('base_fea_size_tr:',fea_tr.shape)
    print('base_fea_size_te:',fea_te.shape)
    
    def data_process_lr(data,meta_data,data_source='train'):
        fea_data = data[['PetID']+num_cols_0+sent_cols_0].copy()
        
        meta_1 = meta_data.query('PicID =="1"')
        temp = data[['PetID']].merge(meta_1,how='left',on='PetID')[meta_cols_0].fillna(0)
        fea_data = pd.concat([fea_data,temp],axis=1)[['PetID']+num_cols_0+sent_cols_0+meta_cols_0]
        
        for i in ['Fee','VideoAmt','PhotoAmt']:
            fea_data[i+'_avg'] = data[i] / data['Quantity']
        
    
        fea_data['color_cnt'] = (data[['Color1','Color2','Color3']] != 0 ).sum(axis=1)
        
        # Description
        temp = data['Description']
        fea_data['desc_count'] = temp.str.count(u'[\u4e00-\u9fa5]') + temp.replace(u'[\u4e00-\u9fa5]','').str.count(' ')
        
        # RescuerID
        fea_data['rescuer_cnt'] = data.groupby('RescuerID')['PetID'].transform('count')
        
        temp = data.groupby(['RescuerID','Type'])['PetID'].count().rename('rescuer_type_cnt').reset_index()
        temp = temp.pivot(index='RescuerID',columns='Type',values='rescuer_type_cnt').fillna(0)
        temp.columns = ['rescuer_type1_cnt','rescuer_type2_cnt']
        temp = data[['RescuerID']].merge(temp,how='left',on='RescuerID')
        fea_data[['rescuer_type1_cnt','rescuer_type2_cnt']] = temp[['rescuer_type1_cnt','rescuer_type2_cnt']] 
        fea_data['rescuer_type1_rate']= fea_data['rescuer_type1_cnt'] / fea_data['rescuer_cnt']
        
      
        # Breed
        temp = data_tr.groupby(['Breed1'])['PetID'].count().rename('Breed1_cnt').reset_index()
        temp = data[['Breed1']].merge(temp,how='left',on='Breed1')   
        fea_data['Breed1_cnt'] = temp['Breed1_cnt']
        
        
        data_tr['Fee_avg'] = data_tr['Fee'] / data_tr['Quantity']
        temp = data_tr.groupby(['Breed1'])['Fee_avg'].mean().rename('Breed1_Fee_avg').reset_index()
        temp = data[['Breed1']].merge(temp,how='left',on='Breed1') 
        fea_data['Breed1_Fee_avg'] = temp['Breed1_Fee_avg']
        fea_data['Breed1_Fee_avg_diff'] = fea_data['Fee_avg'] - fea_data['Breed1_Fee_avg']
        
    
        
        
        
        
        # State
        
        temp = data_tr.groupby(['State','Type'])['PetID'].count().rename('state_type_cnt').reset_index()
        temp = temp.pivot(index='State',columns='Type',values='state_type_cnt').fillna(0)
        # temp.columns = ['state_type1_cnt','state_type2_cnt']
        temp.columns = ['state_type1_cnt_rank','state_type2_cnt_rank']
        temp['state_type1_cnt_rank'] = temp['state_type1_cnt_rank'].rank()
        temp['state_type2_cnt_rank'] = temp['state_type2_cnt_rank'].rank()
        temp = data[['State']].merge(temp,how='left',on='State')
        fea_data[['state_type1_cnt_rank','state_type2_cnt_rank']] = temp[['state_type1_cnt_rank','state_type2_cnt_rank']] 
        
        dummy_fea = pd.get_dummies(data[cat_cols_0],columns=cat_cols_0)
        fea_data = pd.concat([fea_data,dummy_fea],axis=1)   
        fea_data['no_name'] = (data['Name'].isnull() | data['Name'].str.contains('No Name')).fillna(1).astype(int)
        
        
        if data_source == 'train':
            label = data['AdoptionSpeed'].values
        else:
            label = None
        return fea_data,label
    
    lr_fea_tr,label_tr = data_process_lr(data_tr,meta_tr,data_source='train')
    lr_fea_te,_        = data_process_lr(data_te,meta_te,data_source='test')
    
    lr_base_fea_cols = [i for i in lr_fea_te.columns if i in lr_fea_tr.columns]
    lr_fea_tr = lr_fea_tr[lr_base_fea_cols]
    lr_fea_te = lr_fea_te[lr_base_fea_cols]
    replace_cols_0 = [
     'bounding_importance_fracs',
     'dominant_blues',
     'dominant_greens',
     'dominant_reds',
    'label_scores',]
    
    
    lr_fea_tr[replace_cols_0] = lr_fea_tr[replace_cols_0].replace(-1,0)
    lr_fea_te[replace_cols_0] = lr_fea_te[replace_cols_0].replace(-1,0)
    # fillna log
    for i in lr_base_fea_cols[1:35]:
        if i in ['doc_sent_mag','doc_sent_score','Breed1_Fee_avg_diff']:
            lr_fea_tr[i] = lr_fea_tr[i].fillna(0)
            lr_fea_te[i] = lr_fea_te[i].fillna(0)
        else:
            lr_fea_tr[i] = np.log(lr_fea_tr[i].fillna(0)+1e-5)
            lr_fea_te[i] = np.log(lr_fea_te[i].fillna(0)+1e-5)
    # standarscaler
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    lr_fea_tr[lr_base_fea_cols[1:35]] = scaler.fit_transform(lr_fea_tr[lr_base_fea_cols[1:35]])
    lr_fea_te[lr_base_fea_cols[1:35]] = scaler.transform(lr_fea_te[lr_base_fea_cols[1:35]])
    
    
    ##############img_feat###############################
    train_df = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
    pet_ids = train_df['PetID'].values
    n_batches = len(pet_ids) // batch_size + 1
    
    
    def img_model():
        K.clear_session()
        inp = Input((img_size, img_size, 3))
        x = DenseNet121(
                include_top=False, 
                weights="../input/keras-pretrain-model-weights/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5", 
                input_shape=(img_size, img_size, 3))(inp)
        x = GlobalAveragePooling2D()(x)
        x = Lambda(lambda x: K.expand_dims(x, axis = -1))(x)
        x = AveragePooling1D()(x)
        out = Lambda(lambda x: x[:, :, 0])(x)
    
        model = Model(inp, out)
        return model
    
    extract_model = img_model()
    img_features = {}
    for b in tqdm_notebook(range(n_batches)):
        start = b*batch_size
        end = (b+1)*batch_size
        batch_pets = pet_ids[start:end]
        batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
        for i,pet_id in enumerate(batch_pets):
            try:
                batch_images[i] = load_image("../input/petfinder-adoption-prediction/train_images/", pet_id)
            except:
                pass
        batch_preds = extract_model.predict(batch_images)
        for i,pet_id in enumerate(batch_pets):
            img_features[pet_id] = batch_preds[i]
    img_tr = pd.DataFrame.from_dict(img_features, orient='index')
    del img_features
    gc.collect()
    
    pca1 = PCA(n_components=30,random_state=42)
    image_pca_all_tr = pca1.fit_transform(img_tr)
    pca2 = PCA(n_components=80,random_state=42)
    
    image_pca_lr_tr = pca2.fit_transform(img_tr)
    del img_tr
    gc.collect()
    
    test_df = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
    pet_ids = test_df['PetID'].values
    n_batches = len(pet_ids) // batch_size + 1
    
    img_features = {}
    for b in tqdm_notebook(range(n_batches)):
        start = b*batch_size
        end = (b+1)*batch_size
        batch_pets = pet_ids[start:end]
        batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
        for i,pet_id in enumerate(batch_pets):
            try:
                batch_images[i] = load_image("../input/petfinder-adoption-prediction/test_images/", pet_id)
            except:
                pass
        batch_preds = extract_model.predict(batch_images)
        for i,pet_id in enumerate(batch_pets):
            img_features[pet_id] = batch_preds[i]
            
    img_te = pd.DataFrame.from_dict(img_features, orient='index')
    del img_features,train_df,test_df,pet_ids,extract_model
    gc.collect()
    image_pca_all_te = pca1.transform(img_te)
    image_pca_lr_te = pca2.transform(img_te)
    del img_te,pca1,pca2
    gc.collect()
    
    #####################################################
    
    
    
    desc_vec = TfidfVectorizer(ngram_range=(1,4),
                               min_df=3, max_df=0.9, 
                               strip_accents='unicode', 
                               use_idf=1,smooth_idf=1, sublinear_tf=1)
    
    
    desc_tfidf_all_tr = desc_vec.fit_transform(data_tr['Description'].fillna('null').tolist())
    desc_tfidf_all_te = desc_vec.transform(data_te['Description'].fillna('null').tolist())
    
    # tfidf select by chi2 top 10k
    tfidf_select = chi2(desc_tfidf_all_tr,label_tr)
    tfidf_select_index = (-tfidf_select[0]).argsort()[:10000]
    
    desc_tfidf_tr = desc_tfidf_all_tr[:,tfidf_select_index]
    desc_tfidf_te = desc_tfidf_all_te[:,tfidf_select_index]
    
    # SVD
    svd = TruncatedSVD(n_components=200,random_state=42)
    tfidf_svd_tr = svd.fit_transform(desc_tfidf_all_tr)
    tfidf_svd_te = svd.transform(desc_tfidf_all_te)
    
    tfidf_select = chi2(desc_tfidf_all_tr,label_tr)
    tfidf_select_index = (-tfidf_select[0]).argsort()[:50000]
    
    lr_desc_tfidf_tr = desc_tfidf_all_tr[:,tfidf_select_index]
    lr_desc_tfidf_te = desc_tfidf_all_te[:,tfidf_select_index]
    
    join_data_all = pd.concat([data_tr[join_cols_0],data_te[join_cols_0]],axis=0)
    # join_data_all = pd.concat([data_tr[cat_cols_0+join_cols_0],data_te[cat_cols_0+join_cols_0]],axis=0)
    # for i in cat_cols_0:
    #     join_data_all[i] = join_data_all[i].astype(str)+'_'+i
    for i in join_cols_0:
        if i != 'State':
            join_data_all[i] = join_data_all[i].astype(str)+'_'+i[:-1]
        else:
            join_data_all[i] = join_data_all[i].astype(str)+'_'+i
    
    join_data_all = join_data_all.apply(lambda x: ' '.join(x), axis=1).tolist()
    
    # tfidf
    join_vec = TfidfVectorizer(ngram_range=(1,1),
                               min_df=3, max_df=0.9, 
                               strip_accents='unicode', 
                               use_idf=1,smooth_idf=1, sublinear_tf=1)
    # join_all = join_vec.fit_transform(join_data_all)
    join_tr = join_vec.fit_transform(join_data_all[:n_tr])
    join_te = join_vec.transform(join_data_all[n_tr:])
    
    # SVD
    svd = TruncatedSVD(n_components=50,random_state=42)
    # svd_join = svd.fit_transform(join_all)
    
    join_svd_tr = svd.fit_transform(join_tr)
    join_svd_te = svd.transform(join_te)
    
    meta_desc_all = meta_tr[['PetID','label_all_descriptions']].append(meta_te[['PetID','label_all_descriptions']])
    
    meta_desc_all = meta_desc_all.groupby('PetID')['label_all_descriptions']\
                    .apply(lambda x: ' '.join(x)).to_frame().reset_index()
    
    meta_desc_all = data_tr[['PetID']].append(data_te[['PetID']]).merge(meta_desc_all,how='left',on='PetID')['label_all_descriptions']
    
    meta_desc_vec = TfidfVectorizer(ngram_range=(1,1),
                               min_df=10, max_df=0.9, 
                               strip_accents='unicode', 
                               use_idf=1,smooth_idf=1, sublinear_tf=1)
    # meta_desc_tfidf = meta_desc_vec.fit_transform(meta_desc_all.fillna('nan'))
    # meta_desc_tfidf_tr = meta_desc_tfidf[:n_tr,:]
    # meta_desc_tfidf_te = meta_desc_tfidf[n_tr:,:]
    meta_desc_tfidf_tr = meta_desc_vec.fit_transform(meta_desc_all[:n_tr].fillna('nan'))
    meta_desc_tfidf_te = meta_desc_vec.transform(meta_desc_all[n_tr:].fillna('nan'))
     
    del meta_desc_all,join_data_all
    gc.collect()
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    gpf = GroupKFold(n_splits=5)
    
    lr_tr_fea_all = hstack([lr_fea_tr.values[:,1:].astype(float),lr_desc_tfidf_tr,join_tr,meta_desc_tfidf_tr,image_pca_lr_tr]).tocsr()
    lr_te_fea_all = hstack([lr_fea_te.values[:,1:].astype(float),lr_desc_tfidf_te,join_te,meta_desc_tfidf_te,image_pca_lr_te]).tocsr()
    del lr_desc_tfidf_tr,lr_desc_tfidf_te,lr_fea_tr,lr_fea_te,image_pca_lr_tr,image_pca_lr_te
    gc.collect()
    lr_train_predictions = np.zeros((n_tr, 1))
    lr_test_predictions = np.zeros((n_te, 1))
    lr_train = np.zeros((n_tr, ))
    lr_test = np.zeros((n_te, ))
    qwk = []
    rmse_list = []
    for k,(tr_idx,val_idx) in enumerate(gpf.split(data_tr,label_tr,data_tr['RescuerID'])):
    # for k,(tr_idx,val_idx) in enumerate(skf.split(data_tr,label_tr)):
        if k >= 0:
            print('fold_' + str(k) + ' ...')
            
            fold_tr_fea,fold_tr_label = lr_tr_fea_all[tr_idx,:],label_tr[tr_idx]
            fold_val_fea,fold_val_label = lr_tr_fea_all[val_idx,:],label_tr[val_idx]
            
            for i in [2]:
                ridge_model = Ridge(alpha=i,random_state=42)
                ridge_model.fit(fold_tr_fea,fold_tr_label)
    
                pred_val  = ridge_model.predict(fold_val_fea)
    
                optR = OptimizedRounder()
                optR.fit(pred_val, fold_val_label)
                valid_p = optR.predict(pred_val, optR.coefficients(),384).astype(int)  
                qwk_i = cohen_kappa_score(fold_val_label, valid_p,weights='quadratic')
                rmse_i = rmse(fold_val_label,pred_val)
                qwk.append(qwk_i)
                rmse_list.append(rmse_i)
                print('alpha:',i)
                print('qwk:',qwk_i)
                print(optR.coefficients())
                print('train_rmse:',rmse(fold_tr_label,ridge_model.predict(fold_tr_fea)))
                print('val_rmse:',rmse_i)
                print('----'*5)
                
                lr_train_predictions[val_idx] = pred_val.reshape(-1,1)
                lr_train[val_idx] = np.squeeze(pred_val)
                test_preds = ridge_model.predict(lr_te_fea_all)
                lr_test_predictions +=test_preds.reshape(-1,1)
                lr_test += np.squeeze(test_preds)/5
    lr_test_predictions =lr_test_predictions/5
    print(qwk,np.mean(qwk),np.std(qwk))
    print(rmse_list,np.mean(rmse_list),np.std(rmse_list))
    del lr_tr_fea_all,lr_te_fea_all
    gc.collect()
    params = {'application': 'regression',
              'boosting': 'gbdt',
              'metric': 'rmse',
              'num_leaves': 70,
              'max_depth': 8,
              'learning_rate': 0.01,
              'bagging_fraction': 0.85,
              'feature_fraction': 0.8,
              'min_split_gain': 0.02,
              'min_child_samples': 150,
              'min_child_weight': 0.2,
              'lambda_l2': 0.05,
              'verbosity': -1,
              'seed':24}
    
    tr_fea_all = hstack([fea_tr.values[:,1:].astype(float),desc_tfidf_tr,tfidf_svd_tr,join_tr,join_svd_tr,meta_desc_tfidf_tr,image_pca_all_tr,lr_train_predictions,breed_encode_tr]).tocsr()
    te_fea_all = hstack([fea_te.values[:,1:].astype(float),desc_tfidf_te,tfidf_svd_te,join_te,join_svd_te,meta_desc_tfidf_te,image_pca_all_te,lr_test_predictions,breed_encode_te]).tocsr()
    del fea_tr,fea_te,desc_tfidf_tr,tfidf_svd_tr,join_tr,join_svd_tr,meta_desc_tfidf_tr,image_pca_all_tr,desc_tfidf_te,tfidf_svd_te,join_te,join_svd_te,meta_desc_tfidf_te,image_pca_all_te,breed_encode_tr,breed_encode_te
    gc.collect()
    qwk = []
    rmse_list = []
    # fea_imp = []
    train_predictions = np.zeros((n_tr, ))
    test_predictions = np.zeros((n_te, ))
    
    for k,(tr_idx,val_idx) in enumerate(skf.split(data_tr,label_tr)):
    # for k,(tr_idx,val_idx) in enumerate(gpf.split(data_tr,label_tr,data_tr['RescuerID'])):
        if k >= 0:
            print('fold_' + str(k) + ' ...')
            
            
            fold_tr_fea,fold_tr_label = tr_fea_all[tr_idx,:],label_tr[tr_idx]
            fold_val_fea,fold_val_label = tr_fea_all[val_idx,:],label_tr[val_idx]
            
            d_fold_tr = lgb.Dataset(fold_tr_fea, label=fold_tr_label)
            d_fold_val = lgb.Dataset(fold_val_fea, label=fold_val_label)
            watchlist = [d_fold_tr, d_fold_val]
            num_rounds = 10000
            verbose_eval = 100
            early_stop = 100
            model = lgb.train(params,
                              train_set=d_fold_tr,
                              num_boost_round=num_rounds,
                              valid_sets=watchlist,
                              verbose_eval=verbose_eval,
                              early_stopping_rounds=early_stop)    
            pred_val = model.predict(fold_val_fea, num_iteration=model.best_iteration)
            optR = OptimizedRounder()
            optR.fit(pred_val, fold_val_label)
            valid_p = optR.predict(pred_val, optR.coefficients(),384).astype(int)     
            qwk_i = cohen_kappa_score(fold_val_label, valid_p,weights='quadratic')
            rmse_i = model.best_score['valid_1']['rmse']
            qwk.append(qwk_i)
            rmse_list.append(rmse_i)
            print('qwk:',qwk_i)
            print('rmse:',rmse_i)
            print('----'*5)
            
            train_predictions[val_idx] = np.squeeze(pred_val)
            test_preds = model.predict(te_fea_all, num_iteration=model.best_iteration)
            test_predictions += np.squeeze(test_preds)/5
    lgb7_test = [r for r in test_predictions]
    
    lgb7_train = [r for r in train_predictions]
    
    
    del tr_fea_all,te_fea_all 
    del data_tr,meta_tr,data_te,meta_te
    gc.collect()
    
    return lr_train,lr_test,lgb7_train,lgb7_test
lr_train,lr_test,lgb7_train,lgb7_test = feat4_model()
t11=time.time()
print("model11 cost:{} s".format(t11-t10))

import psutil
info = psutil.virtual_memory()
print("memery used rate:",info.percent)

vv=pd.DataFrame(index=range(len(train_data)))
vv['PetID']=train_data['PetID']
vv['lgb1']=lgb1_train
vv['lgb2']=lgb2_train
vv['lgb3']=lgb3_train
vv['lr']=lr_train
vv['lgb5']=lgb5_train
vv['lgb6']=lgb6_train
vv['lgb7']=lgb7_train
vv['cat1']=cat1_train
vv['cat2']=cat2_train
vv['nn1']=nn1_train
vv['nn2']=nn2_train
vv['nn3']=nn3_train
# vv['ridge_pred']=train_data['ridge_pred']
# vv['FTRL_pred']=train_data['FTRL_pred']
# vv['svr_pred']=train_data['svr_pred']
# vv['pac_pred']=train_data['pac_pred']
# vv['sgd_pred']=train_data['sgd_pred']
# vv['breed1_pred']=breed_encode_tr['Breed1_str_pred']
# vv['breed_pred']=breed_encode_tr['breed_pred']
# vv = pd.merge(vv,df_stack1,on='PetID',how="left")
# vv = pd.merge(vv,df_stack2,on='PetID',how="left")
vv = pd.merge(vv,df_stack3,on='PetID',how="left")
vv = pd.merge(vv,df_stack4,on='PetID',how="left")
vv['AdoptionSpeed']=train_data['AdoptionSpeed']

col = [x for x in vv.columns if x not in ['PetID','AdoptionSpeed']]
print(vv[col].corr())

sub=pd.DataFrame(index=range(len(test_data)))
sub['PetID']=test_data['PetID']
sub['lgb1']=lgb1_test
sub['lgb2']=lgb2_test
sub['lgb3']=lgb3_test
sub['lr']=lr_test
sub['lgb5']=lgb5_test
sub['lgb6']=lgb6_test
sub['lgb7']=lgb7_test
sub['cat1']=cat1_test
sub['cat2']=cat2_test
sub['nn1']=nn1_test
sub['nn2']=nn2_test
sub['nn3']=nn3_test
# sub['ridge_pred']=test_data['ridge_pred']
# sub['FTRL_pred']=test_data['FTRL_pred']
# sub['svr_pred']=test_data['svr_pred']
# sub['pac_pred']=test_data['pac_pred']
# sub['sgd_pred']=test_data['sgd_pred']
# sub['breed1_pred']=breed_encode_te['Breed1_str_pred']
# sub['breed_pred']=breed_encode_te['breed_pred']
# sub = pd.merge(sub,df_stack1,on='PetID',how="left")
# sub = pd.merge(sub,df_stack2,on='PetID',how="left")
sub = pd.merge(sub,df_stack3,on='PetID',how="left")
sub = pd.merge(sub,df_stack4,on='PetID',how="left")

print(sub[col].corr())

del train_data,test_data,df_stack3,df_stack4
gc.collect()

train=vv
test=sub

vote = pd.DataFrame(index=range(len(test)))

optR = OptimizedRounder()
train_predictions = np.array(lgb1_train)*0.3+np.array(lgb7_train)*0.3+np.array(lgb3_train)*0.2+np.array(nn3_train)*0.2
blend_train=train_predictions.copy()
boundaries = get_class_bounds(train[label], train_predictions)
optR.fit(train_predictions, train[label],boundaries)
coefficients_ = optR.coefficients()
print(coefficients_)

test_predictions =  np.array(lgb1_test)*0.3+np.array(lgb7_test)*0.3+np.array(lgb3_test)*0.2+np.array(nn3_test)*0.2
blend_test=test_predictions.copy()
test_predictions = optR.predict(test_predictions, coefficients_,90).astype(int)
print(Counter(test_predictions))
vote['blend1']=test_predictions

features = [x for x in train.columns if x not in ['PetID','AdoptionSpeed']]

label='AdoptionSpeed'

params = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 80,
         'max_depth':9,
          'learning_rate': 0.01,
          'bagging_fraction': 0.9,
           "bagging_freq":3,
          'feature_fraction': 0.9,
          'min_split_gain': 0.01,
          'min_child_samples': 150,
          "lambda_l1": 0.1,
          'verbosity': -1,
          'early_stop': 100,
          'verbose_eval': 200,
           "data_random_seed":3,
#           "random_state":1017,
          'num_rounds': 10000}

results = run_cv_model(train[features], test[features], train[label], runLGB, params, rmse, 'LGB')

imports = results['importance'].groupby('feature')['feature', 'importance'].mean().reset_index()
imp=imports.sort_values('importance', ascending=False)
print(imp)

lgb_train = [r[0] for r in results['train']]
lgb_test = [r[0] for r in results['test']]
optR = OptimizedRounder()
train_predictions =lgb_train
boundaries = get_class_bounds(train[label], train_predictions)
optR.fit(train_predictions, train[label],boundaries)
coefficients_ = optR.coefficients()

test_predictions =  lgb_test
test_predictions = optR.predict(test_predictions, coefficients_,90).astype(int)
print(Counter(test_predictions))
vote['lgb']=test_predictions

def runBR(train_X, train_y, test_X, test_y, test_X2, params):


    model = BayesianRidge()
    model.fit(train_X,train_y)
    print('Predict 1/2')
    pred_test_y = model.predict(test_X)
    log=0#log_loss(test_y,pred_test_y)
    print("log_loss:",log)
    class_list=[0,1,2,3,4]
#     pred_test_y=np.array([sum(pred_test_y[ix]*class_list) for
#                                ix in range(len(pred_test_y[:,0]))]) 
    optR = OptimizedRounder()
    optR.fit(pred_test_y, test_y)
    len_0 = sum([1 for i in test_y if i==0])
    coefficients = optR.coefficients()
    pred_test_y_k = optR.predict(pred_test_y, coefficients,len_0)
   
    print("Valid Counts = ", Counter(test_y))
    print("Predicted Counts = ", Counter(pred_test_y_k))
    print("Coefficients = ", coefficients)
    qwk = cohen_kappa_score(test_y, pred_test_y_k,weights='quadratic')
    print("QWK = ", qwk)
    print('Predict 2/2')
    pred_test_y2 = model.predict(test_X2)
#     pred_test_y2=np.array([sum(pred_test_y2[ix]*class_list) for
#                                ix in range(len(pred_test_y2[:,0]))]) 
    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), 0, coefficients, qwk,log
results = run_cv_model(train[features], test[features], train[label], runBR, params, rmse, 'LR')

br_train = [r[0] for r in results['train']]
br_test = [r[0] for r in results['test']]
optR = OptimizedRounder()
train_predictions =br_train
boundaries = get_class_bounds(train[label], train_predictions)
optR.fit(train_predictions, train[label],boundaries)
coefficients_ = optR.coefficients()

test_predictions =  br_test
test_predictions = optR.predict(test_predictions, coefficients_,90).astype(int)
print(Counter(test_predictions))
vote['br']=test_predictions

params={
	'booster':'gbtree',
	'objective': 'reg:linear',
    "tree_method":"gpu_hist",
            "gpu_id":0,
#      'is_unbalance':'True',
# 	'scale_pos_weight': 1500.0/13458.0,
        'eval_metric': "rmse",
	'gamma':0.2,#0.2 is ok
	'max_depth':7,
# 	'lambda':20,
    # "alpha":5,
        'subsample':0.9,
        'colsample_bytree':0.9 ,
        'min_child_weight':3, 
        'eta': 0.01,
    # 'learning_rate':0.01,
    "silent":1,
	'seed':1024,
	'nthread':12,
     'num_rounds': 5000,
    'verbose_eval': 200,
    'early_stop':100,

   
    }
def runXGB(train_X, train_y, test_X, test_y, test_X2, params):
#     print('Prep LGB')
    d_train = xgb.DMatrix(train_X, label=train_y)
    d_valid = xgb.DMatrix(test_X, label=test_y)
    watchlist = [(d_train,'train'),
    (d_valid,'val')
             ]
#     print('Train LGB')
    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    early_stop = None
    if params.get('early_stop'):
        early_stop = params.pop('early_stop')
    model = xgb.train(params,
                      d_train,
                      num_boost_round=num_rounds,
                      evals=watchlist,
#                       fobj=softkappaObj,
                      verbose_eval=verbose_eval,
#                       feval=kappa_scorer,
                      early_stopping_rounds=early_stop)
    print('Predict 1/2')
    pred_test_y = model.predict(xgb.DMatrix(test_X),ntree_limit=model.best_ntree_limit)
    log=0#log_loss(test_y,pred_test_y)
    print("log_loss:",log)
    class_list=[0,1,2,3,4]
#     pred_test_y=np.array([sum(pred_test_y[ix]*class_list) for
#                                ix in range(len(pred_test_y[:,0]))]) 
    optR = OptimizedRounder()
    optR.fit(pred_test_y, test_y)
    len_0 = sum([1 for i in test_y if i==0])
    coefficients = optR.coefficients()
    pred_test_y_k = optR.predict(pred_test_y, coefficients,len_0)
   
    print("Valid Counts = ", Counter(test_y))
    print("Predicted Counts = ", Counter(pred_test_y_k))
    print("Coefficients = ", coefficients)
    qwk = cohen_kappa_score(test_y, pred_test_y_k,weights='quadratic')
    print("QWK = ", qwk)
    print('Predict 2/2')
    pred_test_y2 = model.predict(xgb.DMatrix(test_X2),ntree_limit=model.best_ntree_limit)
#     pred_test_y2=np.array([sum(pred_test_y2[ix]*class_list) for
#                                ix in range(len(pred_test_y2[:,0]))]) 
   
    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), model.get_fscore(), coefficients, qwk,log
results = run_cv_model(train[features], test[features], train[label], runXGB, params, rmse, 'XGB')

xgb_train = [r[0] for r in results['train']]
xgb_test = [r[0] for r in results['test']]
optR = OptimizedRounder()
train_predictions =xgb_train
boundaries = get_class_bounds(train[label], train_predictions)
optR.fit(train_predictions, train[label],boundaries)
coefficients_ = optR.coefficients()

test_predictions =  xgb_test
test_predictions = optR.predict(test_predictions, coefficients_,90).astype(int)
print(Counter(test_predictions))
vote['xgb']=test_predictions


results = run_cv_model(train[features], test[features], train[label], runCAT, params, rmse, 'CAT')
cat_train = [r[0] for r in results['train']]
cat_test = [r[0] for r in results['test']]

optR = OptimizedRounder()
train_predictions =cat_train
boundaries = get_class_bounds(train[label], train_predictions)
optR.fit(train_predictions, train[label],boundaries)
coefficients_ = optR.coefficients()

test_predictions =  cat_test
test_predictions = optR.predict(test_predictions, coefficients_,90).astype(int)
print(Counter(test_predictions))
vote['cat']=test_predictions

optR = OptimizedRounder()
train_predictions = (np.array(lgb_train)+np.array(xgb_train)+np.array(cat_train)+np.array(br_train))/4.0#[r[0] for r in results['train']]##np.array(lgb1_train)*0.7+np.array(lgb2_train)*0.3
stack_train = train_predictions.copy()
boundaries = get_class_bounds(train[label], train_predictions)
optR.fit(train_predictions, train[label],boundaries)
coefficients_ = optR.coefficients()
print(coefficients_)

test_predictions =  (np.array(lgb_test)+np.array(xgb_test)+np.array(cat_test)+np.array(br_test))/4.0#[r[0] for r in results['test']]#reg.coef_[0] *sub["pred1"]+reg.coef_[1] *sub["pred2"]#np.array(lgb1_test)*0.7+np.array(lgb2_test)*0.3
stack_test = test_predictions.copy()
test_predictions = optR.predict(test_predictions, coefficients_,90).astype(int)
print(Counter(test_predictions))
vote['blend2']=test_predictions

####blend+stacking
optR = OptimizedRounder()
train_predictions = np.array(blend_train)*0.5+np.array(stack_train)*0.5#[r[0] for r in results['train']]##np.array(lgb1_train)*0.7+np.array(lgb2_train)*0.3
boundaries = get_class_bounds(train[label], train_predictions)
optR.fit(train_predictions, train[label],boundaries)
coefficients_ = optR.coefficients()
print(coefficients_)

test_predictions =  np.array(blend_test)*0.5+np.array(stack_test)*0.5#[r[0] for r in results['test']]#reg.coef_[0] *sub["pred1"]+reg.coef_[1] *sub["pred2"]#np.array(lgb1_test)*0.7+np.array(lgb2_test)*0.3
test_predictions = optR.predict(test_predictions, coefficients_,90).astype(int)
print(Counter(test_predictions))
vote['blend+stack']=test_predictions
vote['blend+stack_copy']=test_predictions

test_predictions = vote.mode(1)[0].values.astype(int)
test_predictions  = vote['br'].values
train_predictions = optR.predict(train_predictions, coefficients_,410).astype(int)
print(cohen_kappa_score(train[label], train_predictions,weights='quadratic'))
print(rmse(train[label], train_predictions))
submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions})
print(submission.head())
print(submission.AdoptionSpeed.value_counts())
submission.to_csv('submission.csv', index=False)
print("done! cost:{} s".format(time.time()-start_time))
    
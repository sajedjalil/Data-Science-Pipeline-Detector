IS_LOCAL = False
IS_KAGGLE_SERVER = not IS_LOCAL
if IS_KAGGLE_SERVER:
    import os
    os.environ['NUMEXPR_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    import mkl
    mkl.set_num_threads(4)

import enum
import gc
import time
from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy as scipy
import scipy.sparse as sp
from joblib import Parallel, delayed
from keras.callbacks import LearningRateScheduler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class Timer:
    def __init__(self, timer_name):
        self.timer_name = timer_name
        self.start_time = self.last_time = time.time()

    def get_total_and_since_last(self):
        current_time = time.time()
        total = current_time - self.start_time
        since_last = current_time - self.last_time
        self.last_time = current_time
        return total, since_last

    def print(self, message):
        total, since_last = self.get_total_and_since_last()
        print('### [{}] [{:6.1f}] [{:6.1f}] {}'.format(self.timer_name, total, since_last, message))


def rmse(y, correct_y):
    return np.sqrt(mean_squared_error(y, correct_y))


def split_category_name(category_name: str):
    spl = category_name.split('/', 2)
    if len(spl) == 3:
        return spl
    else:
        return ['', '', '']


def read_dataset(down_sampling_rate=None, train_size=None):
    train = pd.read_table('../input/train.tsv', engine='c')
    test = pd.read_table('../input/test.tsv', engine='c')
    if down_sampling_rate is not None:
        assert 0 < down_sampling_rate < 1.0
        train = train.sample(frac=down_sampling_rate, random_state=1919)
        test = test.sample(frac=down_sampling_rate, random_state=810)

    train['labeled'] = True
    train['log1p_price'] = np.log1p(train['price'])

    train['for_train'] = True
    if train_size is not None:
        assert 0 < train_size < 1.0
        train = train.sample(frac=1, random_state=364364).reset_index(drop=True)
        train.loc[int(train.shape[0] * train_size):, 'for_train'] = False

    test['labeled'] = False
    test['for_train'] = False
    train.rename(columns={'train_id': 'id'}, inplace=True)
    test.rename(columns={'test_id': 'id'}, inplace=True)

    dataset = pd.concat([train, test], ignore_index=True)

    clean_dirty_data(dataset)

    # extend
    dataset['name'] = dataset['brand_name'] + ' ' + dataset['name']
    dataset['item_description'] = dataset['name'] + ' ' + \
                                  dataset['item_description']

    # preprocessor = Preprocessor().fit(dataset['name'].tolist() + dataset['item_description'].tolist())
    # dataset['name'] = dataset['name'].map(preprocessor.preprocess_symbols)
    # dataset['item_description'] = dataset['item_description'].map(preprocessor.preprocess_symbols)

    add_sub_cagegories(dataset)

    return dataset


def clean_dirty_data(dataset):
    dataset['name'].fillna('', inplace=True)
    dataset['category_name'].fillna('', inplace=True)
    dataset['brand_name'].fillna('', inplace=True)
    dataset['item_description'].fillna('', inplace=True)

    dataset.drop(dataset[dataset.price < 1.0].index, inplace=True)
    dataset.reset_index(inplace=True, drop=True)


def add_sub_cagegories(dataset):
    spl = dataset['category_name'].apply(split_category_name)
    dataset['category1'], dataset['category2'], dataset['category3'] = zip(*spl)


def create_sub_categories(dataset: pd.DataFrame):
    spl = dataset['category_name'].apply(split_category_name)

    df = pd.DataFrame()
    df['category1'], df['category2'], df['category3'] = zip(*spl)

    df['category1'] = df['category1'] + '_' + dataset['item_condition_id'].astype(str)
    df['category2'] = df['category2'] + '_' + dataset['item_condition_id'].astype(str)
    df['category3'] = df['category3'] + '_' + dataset['item_condition_id'].astype(str)

    feature_names = ['category1', 'category2', 'category3']

    n = dataset.shape[0]
    feature = np.concatenate([
        LabelEncoder().fit_transform(df['category1']).reshape(n, 1),
        LabelEncoder().fit_transform(df['category2']).reshape(n, 1),
        LabelEncoder().fit_transform(df['category3']).reshape(n, 1),
    ], axis=1)

    assert feature.shape[0] == n
    assert feature.shape[1] == 3
    return Feature(feature_names, feature, FeatureType.CATEGORICAL)


def descritize_each_column(a, num_bins):
    a = a.copy()
    for col in range(a.shape[1]):
        column = a[:, col]
        _, bins = pd.qcut(np.unique(column), num_bins - 1, duplicates='drop', labels=False, retbins=True)

        descritized_vals = np.digitize(column, bins)
        descritized_vals = LabelEncoder().fit_transform(descritized_vals)

        a[:, col] = descritized_vals
    return a


class FeatureType(enum.Enum):
    NORMAL = enum.auto()
    CATEGORICAL = enum.auto()


class Feature:
    def __init__(self, feature_names, feature, feature_type=FeatureType.NORMAL, bins=None):
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        # assert feature.shape[1] == len(feature_names)
        self.feature_names = feature_names
        self.feature = feature
        self.feature_type = feature_type
        self.bins = bins

    def get_lgb_feature_names(self):
        return self.feature_names

    def get_lgb_features(self):
        return self.feature

    def get_one_hot_like_features(self):
        if self.feature_type == FeatureType.CATEGORICAL:
            assert self.bins is None
            return OneHotEncoder().fit_transform(self.feature)
        elif self.bins is not None:
            # try-catch?
            return OneHotEncoder().fit_transform(descritize_each_column(self.feature, self.bins))
        else:
            return self.feature

    def get_keras_features(self):
        return self.feature

    def get_feature_type(self):
        return self.feature_type


def create_categorical_name_feature(dataset):
    name_df = pd.DataFrame()
    name_df['name'] = dataset['name'].str.lower()
    count_name = name_df['name'].value_counts()
    name_df['count'] = name_df['name'].map(count_name)
    name_df.loc[(name_df['count'] < 2), 'name'] = ''

    name_feature = LabelEncoder().fit_transform(name_df['name']).reshape(-1, 1)

    assert name_feature.shape[0] == dataset.shape[0]
    return Feature('name', name_feature, FeatureType.CATEGORICAL)


def create_features(dataset):
    timer = Timer('feature_creation')
    n = dataset.shape[0]
    features = []

    features.append(Feature('item_condition_id', dataset['item_condition_id'].values.reshape(n, 1), FeatureType.CATEGORICAL))
    features.append(Feature('shipping', dataset['shipping'].values.reshape(n, 1), FeatureType.CATEGORICAL))
    features.append(Feature('brand_name', LabelEncoder().fit_transform(dataset['brand_name']).reshape(n, 1), FeatureType.CATEGORICAL))
    features.append(Feature('category_name', LabelEncoder().fit_transform(dataset['category_name']).reshape(n, 1), FeatureType.CATEGORICAL))
    features.append(create_sub_categories(dataset))
    features.append(create_categorical_name_feature(dataset))

    features.append(Feature('name_len', dataset['name'].str.len().values.reshape(n, 1), bins=10))
    features.append(Feature('item_description_len', dataset['item_description'].str.len().values.reshape(n, 1), bins=40))

    features.append(Feature('name_count_words', dataset['name'].str.count(' ').values.reshape(n, 1) + 1, feature_type=FeatureType.CATEGORICAL))
    features.append(Feature('item_description_count_words', dataset['item_description'].str.count(' ').values.reshape(n, 1) + 1, bins=20))
    timer.print('Done basic')

    return features


def create_sparse_features(dataset, feature_params):
    timer = Timer('sparse feature')
    timer.print('Start')

    sparse_features = []
    sparse_features.append(create_doc_term_of_name(dataset, **feature_params['name_term']))
    timer.print('Done create_doc_term_of_name')

    sparse_features.append(create_doc_term_of_description(dataset, **feature_params['desc_term']))
    timer.print('Done create_doc_term_of_description')

    return sparse_features


def predict(models, features):
    predicts = np.ndarray(shape=(features.shape[0], len(models)), dtype=np.float64)
    for i, model in enumerate(models):
        predicts[:, i] = model.predict(features)
    return predicts.mean(axis=1)


# one hot
def create_one_hot_like_features(features: List[Feature], start, end, train_size, verbose=False):
    one_hot_like_features = []
    for feature in features:
        gc.collect()
        one_hot_like_feature = feature.get_one_hot_like_features()
        one_hot_like_features.append(one_hot_like_feature)
        if verbose:
            print('{:20s}, {}'.format(feature.get_lgb_feature_names()[0], one_hot_like_feature.shape))
    gc.collect()
    f = scipy.sparse.hstack(one_hot_like_features, dtype=np.float64).tocsr()
    f = drop_non_intersect_features(f, train_size)
    # assert np.unique(f.data) == [1]
    f = f[start:end]
    gc.collect()
    return f


def preprocess(a):
    return a.lower().replace('$', ' dolloars ').replace('gb', ' gb ').replace('+', ' plus ')


def drop_non_intersect_features(matrix, train_size):
    intersect = (matrix[:train_size].sum(axis=0) > 0).A1 & (matrix[train_size:].sum(axis=0) > 0).A1
    return matrix[:, intersect]


def create_doc_term_of_name(dataset, min_df, ngram_range):
    name = dataset['name'].map(preprocess)

    vec = CountVectorizer(
        min_df=min_df,
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b\w+\b",
        stop_words='english',
        binary=True
    )
    print(vec)
    count_matrix = vec.fit_transform(name[dataset['for_train']])
    count_matrix = sp.vstack([count_matrix, vec.transform(name[~dataset['for_train']])])

    print(count_matrix.shape)
    count_matrix = drop_non_intersect_features(count_matrix, dataset['for_train'].sum())
    print(count_matrix.shape)

    # assert np.unique(count_matrix.data) == [1]
    return Feature('doc_term_of_name', count_matrix)


def create_doc_term_of_description(dataset, min_df, ngram_range):
    item_description = dataset['item_description'].map(preprocess)

    vec = CountVectorizer(
        min_df=min_df,
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b\w+\b",
        stop_words='english',
        binary=True
    )
    print(vec)
    count_matrix = vec.fit_transform(item_description[dataset['for_train']])
    count_matrix = sp.vstack([count_matrix, vec.transform(item_description[~dataset['for_train']])])

    print(count_matrix.shape)
    count_matrix = drop_non_intersect_features(count_matrix, dataset['for_train'].sum())
    print(count_matrix.shape)

    # assert np.unique(count_matrix.data) == [1]
    return Feature('doc_term_of_description', count_matrix)


def calculate_each_row_mean(a):
    row_sum = np.asarray(a.sum(axis=1)).reshape(-1)
    row_nonzeros = a.getnnz(axis=1)
    row_mean = np.divide(row_sum, row_nonzeros,
                                out=np.full(shape=row_sum.shape, fill_value=np.nan),
                                where=(row_nonzeros > 0))
    return row_mean


def create_ngram_price_mean_(dataset, column, min_df, max_df, ngram_range):
    timer = Timer(column + '_ngram_price_mean')
    timer.print('Start')

    vec = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b\w+\b",
        stop_words='english',
        binary=True
    )
    texts = dataset[column].map(preprocess)
    count_matrix = vec.fit_transform(texts[dataset['for_train']])
    timer.print('Fitted')
    all_count_matrix = vec.transform(texts)
    timer.print('Transformed')

    term_df = np.asarray(count_matrix.sum(axis=0)).reshape(-1)
    assert term_df.shape[0] == count_matrix.shape[1]

    log1p_price = dataset.loc[dataset['for_train'], 'log1p_price'].values
    log1p_price_matrix = count_matrix.multiply(log1p_price.reshape(-1, 1))
    term_log1p_price_mean = np.asarray(log1p_price_matrix.sum(axis=0)).reshape(-1) / term_df

    term_log1p_price_mean_squared = np.asarray(log1p_price_matrix.power(2).sum(axis=0)).reshape(-1) / term_df
    term_log1p_price_var = np.maximum(1e-6, term_log1p_price_mean_squared - term_log1p_price_mean ** 2)

    features_df = pd.DataFrame()
    bins = 20
    _, seps = pd.qcut(term_log1p_price_var, bins, duplicates='drop', labels=False, retbins=True)
    for i in range(len(seps) - 1):
        lower, upper = seps[i], seps[i + 1]
        selector = (lower <= term_log1p_price_var) & (term_log1p_price_var < upper)
        #         print('[lower, upper): {:4.3f} {:4.3f}'.format(lower, upper))

        #
        selected_term_mean = np.where(selector, term_log1p_price_mean, 0)
        selected_mean_matrix = all_count_matrix.multiply(scipy.sparse.csc_matrix(selected_term_mean))
        price_mean_mean = calculate_each_row_mean(selected_mean_matrix)

        # #
        # terms = selected_mean_matrix.getnnz(axis=1)
        #
        # #
        # price_mean_var = calculate_each_row_mean(selected_mean_matrix.power(2)) - price_mean_mean ** 2

        #
        selected_term_df = np.where(selector, term_df, 0)
        selected_df_matrix = all_count_matrix.multiply(scipy.sparse.csc_matrix(selected_term_df))
        df_mean = calculate_each_row_mean(selected_df_matrix)

        features_df['{}_price_mean_mean_{}'.format(column, i)] = price_mean_mean
        # features_df['{}_price_mean_var_{}'.format(column, i)] = price_mean_var
        # features_df['{}_terms_{}'.format(column, i)] = terms
        # features_df['{}_df_mean_{}'.format(column, i)] = df_mean

    timer.print('Done')
    return features_df


def create_ngram_price_mean(dataset, column, min_df, max_df, ngram_range):

    features_df = create_ngram_price_mean_(dataset, column, min_df, max_df, ngram_range)

    feature_names = []
    features = []
    for c in features_df.columns:
        feature_names.append(c)
        features.append(features_df[c].values.reshape(-1, 1))

    # nan or 0 ??
    f = scipy.sparse.csr_matrix(np.hstack(features))

    assert f.shape == (dataset.shape[0], len(feature_names))
    return Feature(feature_names, f)


# TODO tol, alpha調整
def train_ridge_model(train_X, train_y):
    ridge = Ridge(alpha=16.8, solver='sag', copy_X=False, fit_intercept=True, tol=0.01, max_iter=100,
                  random_state=114514)
    ridge.fit(train_X, train_y)
    return ridge


def do_ridge_global(train_index):
    global g_all_train_X, g_all_train_y
    model = train_ridge_model(g_all_train_X[train_index], g_all_train_y[train_index])
    return model


def create_ridge_predict_stacking_feature(dataset, common_features: List[Feature]):
    timer = Timer('ridge_predict')
    timer.print('Start')
    gc.collect()

    train_size = dataset[dataset['for_train']].shape[0]

    train_one_hot_like_features = create_one_hot_like_features(common_features, 0, train_size, train_size, verbose=True)
    gc.collect()
    assert train_one_hot_like_features.shape[0] == train_size
    print(train_one_hot_like_features.shape)
    timer.print('Created one hot like features for train')

    global g_all_train_X, g_all_train_y
    g_all_train_X = train_one_hot_like_features
    g_all_train_y = dataset['log1p_price'][:train_size].values

    gc.collect()
    kf = KFold(n_splits=8, shuffle=True, random_state=810)
    models = Parallel(n_jobs=4, max_nbytes=None)(
        delayed(do_ridge_global)(train_index)
        for train_index, _ in kf.split(train_one_hot_like_features)
    )
    del g_all_train_X, g_all_train_y
    gc.collect()
    timer.print('Trained')

    preds = np.full(dataset.shape[0], -114514, dtype=np.float64)
    for i, (train_index, valid_index) in enumerate(kf.split(train_one_hot_like_features)):
        preds[valid_index] = models[i].predict(train_one_hot_like_features[valid_index])
        gc.collect()
    del train_one_hot_like_features
    gc.collect()
    timer.print('Predicted valid')

    test_size = dataset.shape[0] - train_size
    test_batches = 5
    test_batch_size = test_size // test_batches + 1
    for batch_i in range(test_batches):
        gc.collect()
        start = train_size + test_batch_size * batch_i
        end = train_size + min(test_size, test_batch_size * (batch_i + 1))
        test_X = create_one_hot_like_features(common_features, start, end, train_size)

        predss = []
        for model in models:
            predss.append(model.predict(test_X))
        preds[start:end] = np.mean(np.hstack([preds.reshape(-1, 1) for preds in predss]), axis=1)
    gc.collect()
    timer.print('Predicted test')

    print('total loss: ', rmse(preds[:train_size], dataset['log1p_price'][:train_size].values))
    timer.print('Trained ridge models')

    return Feature('ridge_pred', preds.reshape(dataset.shape[0], 1))


import copy
import fastFM.als


def do_single_fm(train_X, train_y, valid_X, valid_y, l2_reg_V=600):
    train_i = 0
    timer = Timer('fm_{}'.format(train_i))
    timer.print('Start')

    model = fastFM.als.FMRegression(n_iter=0, init_stdev=0.0001, rank=8, l2_reg_w=20, l2_reg_V=l2_reg_V)
    model.fit(train_X, train_y)
    train_loss = rmse(model.predict(train_X), train_y)
    valid_loss = rmse(model.predict(valid_X), valid_y)

    train_losses = [train_loss]
    valid_losses = [valid_loss]
    for iter_i in range(10):
        prev_model = copy.deepcopy(model)

        model.fit(train_X, train_y, n_more_iter=1)
        timer.print('Fitted')

        train_loss = rmse(model.predict(train_X), train_y)
        valid_loss = rmse(model.predict(valid_X), valid_y)
        timer.print('Predicted')

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        timer.print('loss {} / iter {:2d}: {:7.5f}, {:7.5f}'.format(train_i, iter_i, train_loss, valid_loss))

        if valid_loss > valid_losses[-2]:
            model = prev_model
            break

    train_loss = rmse(model.predict(train_X), train_y)
    valid_loss = rmse(model.predict(valid_X), valid_y)
    print('loss {}: {}, {}'.format(train_i, train_loss, valid_loss))
    timer.print('Done')

    return model


def create_fm_predict_stacking_feature(dataset, common_features: List[Feature]):
    timer = Timer('fm_predict')
    timer.print('Start')

    train_size = dataset[dataset['for_train']].shape[0]

    one_hot_like_features = create_one_hot_like_features(common_features, 0, dataset.shape[0], train_size)
    assert one_hot_like_features.shape[0] == dataset.shape[0]
    print(one_hot_like_features.shape)
    timer.print('Created one hot like features')

    all_train_X = one_hot_like_features[:train_size]
    all_train_y = dataset['log1p_price'][:train_size].values

    models = []
    preds = np.full(dataset.shape[0], -114514, dtype=np.float64)
    kf = KFold(n_splits=8, shuffle=True, random_state=810)
    for train_i, (train_index, valid_index) in enumerate(kf.split(all_train_X)):
        train_X, valid_X = all_train_X[train_index], all_train_X[valid_index]
        train_y, valid_y = all_train_y[train_index], all_train_y[valid_index]

        model = do_single_fm(train_X, train_y, valid_X, valid_y)

        valid_preds = model.predict(valid_X)
        preds[valid_index] = valid_preds

        print('loss {}:'.format(train_i), rmse(valid_preds, valid_y))

        models.append(model)

        timer.print('Trained fm model {}'.format(train_i))

    print('total loss: ', rmse(preds[:train_size], all_train_y))

    test_X = one_hot_like_features[train_size:]
    preds[train_size:] = np.mean(np.hstack([model.predict(test_X).reshape(test_X.shape[0], 1) for model in models]), axis=1)
    timer.print('Trained fm models')

    return Feature('fm_pred', preds.reshape(dataset.shape[0], 1))


import keras
import keras.preprocessing.text
from keras.layers import Conv1D, GlobalMaxPooling1D, BatchNormalization
from keras.layers import Input, Embedding, Dropout, Dense, Flatten
from keras import backend as K
import tensorflow as tf

maxlen_desc = 100
maxlen_name = 13


def build_cnn(
    num_words, maxlen_name, maxlen_desc, num_categories1, num_categories2, num_categories3,
              num_brand_names, num_item_conditions, additional_feature_names,
    hyper_params
):
    input_desc = Input((maxlen_desc,), name='desc')
    embedding_layer_desc = Embedding(num_words, hyper_params['desc_embedding'])
    embedding_desc = embedding_layer_desc(input_desc)

    input_name = Input((maxlen_name,), name='name')
    embedding_layer_name = Embedding(num_words, hyper_params['name_embedding'])
    embedding_name = embedding_layer_name(input_name)

    input_category1 = Input((1,), name='category1')
    input_category2 = Input((1,), name='category2')
    input_category3 = Input((1,), name='category3')
    embedding_category1 = Embedding(num_categories1, hyper_params['category1_embedding'])(input_category1)
    embedding_category2 = Embedding(num_categories2, hyper_params['category2_embedding'])(input_category2)
    embedding_category3 = Embedding(num_categories3, hyper_params['category3_embedding'])(input_category3)

    input_brand_name = Input((1,), name='brand_name')
    embedding_brand_name = Embedding(num_brand_names, hyper_params['brand_name_embedding'])(input_brand_name)

    input_item_condition = Input((1,), name='item_condition')
    embedding_item_condition = Embedding(num_item_conditions, 2)(input_item_condition)

    input_shipping = Input((1,), name='shipping')
    embedding_shipping = Embedding(2, 1)(input_shipping)

    cnn_desc = Conv1D(filters=hyper_params['desc_filters'], kernel_size=3, activation='relu')(embedding_desc)
    cnn_desc = GlobalMaxPooling1D()(cnn_desc)
    dense_desc = cnn_desc

    cnn_name = Conv1D(filters=hyper_params['name_filters'], kernel_size=3, activation='relu')(embedding_name)
    cnn_name = GlobalMaxPooling1D()(cnn_name)
    dense_name = cnn_name

    numerical_feature_names = [
        'name_len',
        'desc_len',
        'name_count_words',
        'desc_count_words',
        'category1_mean',
        'category2_mean',

        *additional_feature_names
    ]
    numerical_feature_inputs = [Input((1,), name=feature_name) for feature_name in numerical_feature_names]

    x = keras.layers.concatenate([
        dense_desc,
        dense_name,
        Flatten()(embedding_category1),
        Flatten()(embedding_category2),
        Flatten()(embedding_category3),
        Flatten()(embedding_brand_name),
        Flatten()(embedding_item_condition),
        Flatten()(embedding_shipping),

        *numerical_feature_inputs
    ])

    x = BatchNormalization()(x)
    x = Dropout(hyper_params['dropout0'])(Dense(hyper_params['dense0'], activation='relu')(x))
    x = BatchNormalization()(x)
    x = Dropout(hyper_params['dropout1'])(Dense(hyper_params['dense1'], activation='relu')(x))
    x = BatchNormalization()(x)
    output = Dense(1, activation='linear')(x)

    model = keras.Model(
        [
            input_desc,
            input_name,
            input_category1,
            input_category2,
            input_category3,
            input_brand_name,
            input_item_condition,
            input_shipping,

            *numerical_feature_inputs
        ],
        output)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam())
    return model


def to_seqs(tokenizer, texts, maxlen):
    return keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=maxlen,
                                                      truncating='post')


def convert_to_input(tokenizer, dataset, additional_feature_dict):
    timer = Timer('convert_to_input')
    timer.print('Start')

    batches = 4
    batch_size = dataset.shape[0] // batches + 1

    name = np.concatenate(
        Parallel(n_jobs=4)(
            delayed(to_seqs)(tokenizer, name_batch, maxlen_name)
            for name_batch in [dataset.name[batch_size * i:batch_size * (i + 1)].values for i in range(batches)]
        )
    )
    gc.collect()
    timer.print('Done name')

    desc = np.concatenate(
        Parallel(n_jobs=4)(
            delayed(to_seqs)(tokenizer, desc_batch, maxlen_desc)
            for desc_batch in [dataset.item_description[batch_size * i:batch_size * (i + 1)].values for i in range(batches)]
        )
    )
    gc.collect()
    timer.print('Done desc')

    category1 = LabelEncoder().fit_transform(dataset['category1']).reshape(dataset.shape[0], 1)
    category2 = LabelEncoder().fit_transform(dataset['category2']).reshape(dataset.shape[0], 1)
    category3 = LabelEncoder().fit_transform(dataset['category3']).reshape(dataset.shape[0], 1)

    brand_name = LabelEncoder().fit_transform(dataset['brand_name']).reshape(dataset.shape[0], 1)

    item_condition = LabelEncoder().fit_transform(dataset['item_condition_id']).reshape(dataset.shape[0], 1)

    shipping = LabelEncoder().fit_transform(dataset['shipping']).reshape(dataset.shape[0], 1)

    name_len = StandardScaler().fit_transform(dataset['name'].str.len().values.reshape(-1, 1))
    desc_len = StandardScaler().fit_transform(dataset['item_description'].str.len().values.reshape(-1, 1))
    name_count_words = StandardScaler().fit_transform(dataset['name'].str.count(' ').values.reshape(-1, 1))
    desc_count_words = StandardScaler().fit_transform(dataset['item_description'].str.count(' ').values.reshape(-1, 1))

    category1_mean = dataset['category1'].map(dataset.groupby('category1')['log1p_price'].mean()).values.reshape(-1, 1)
    category2_mean = dataset['category2'].map(dataset.groupby('category2')['log1p_price'].mean()).values.reshape(-1, 1)

    timer.print('Almost done')

    keras_input = {
        'name': name,
        'desc': desc,
        'category1': category1,
        'category2': category2,
        'category3': category3,
        'brand_name': brand_name,
        'item_condition': item_condition,
        'shipping': shipping,

        'name_len': name_len,
        'desc_len': desc_len,
        'name_count_words': name_count_words,
        'desc_count_words': desc_count_words,

        'category1_mean': category1_mean,
        'category2_mean': category2_mean,

        **additional_feature_dict
    }
    print(list(keras_input.keys()))
    for vals in keras_input.values():
        assert vals.shape[0] == dataset.shape[0]
    return keras_input


def create_keras_input_etc(dataset):
    timer = Timer('keras')

    # TODO: only train?
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(
        dataset[dataset['for_train']]['item_description'].tolist() +
        dataset[dataset['for_train']]['name'].tolist()
    )
    gc.collect()
    timer.print('Tokenized')

    num_words = len(tokenizer.word_index) + 1  # +1 is for padding
    num_brand_names = dataset['brand_name'].unique().size
    num_item_conditions = dataset['item_condition_id'].unique().size

    num_categories1 = dataset['category1'].unique().size
    num_categories2 = dataset['category2'].unique().size
    num_categories3 = dataset['category3'].unique().size

    timer.print('Prepared dataset')

    ##
    f = create_ngram_price_mean_(dataset, 'name', min_df=30, max_df=10000, ngram_range=(1, 3))
    f.fillna(dataset['log1p_price'].mean(), inplace=True)

    features_dict = {}
    for c in f.columns:
        features_dict[c] = StandardScaler().fit_transform(f[c].values.reshape(-1, 1)).reshape(-1, 1)
    timer.print('Created additional features')

    keras_input = convert_to_input(tokenizer, dataset, additional_feature_dict=features_dict)
    timer.print('Converted to keras input')

    info_to_build_nn = {
        'num_words': num_words,
        'maxlen_name': maxlen_name,
        'maxlen_desc': maxlen_desc,
        'num_categories1': num_categories1,
        'num_categories2': num_categories2,
        'num_categories3': num_categories3,
        'num_brand_names': num_brand_names,
        'num_item_conditions': num_item_conditions,
        'additional_feature_names': list(features_dict.keys()),
    }

    return {
        'keras_input': keras_input,
        'info_to_build_nn': info_to_build_nn,
    }


def do_keras(dataset):
    keras_input_etc = create_keras_input_etc(dataset)
    keras_input = keras_input_etc['keras_input']
    info_to_build_nn = keras_input_etc['info_to_build_nn']

    def slice_keras_input(keras_input, begin, end):
        return {input_name: data[begin:end] for input_name, data in keras_input.items()}

    train_size = dataset[dataset['for_train']].shape[0]
    valid_size = dataset[dataset['labeled'] & ~dataset['for_train']].shape[0]

    train_y = dataset['log1p_price'][:train_size].values
    valid_y = dataset['log1p_price'][train_size:train_size + valid_size]

    train_input = slice_keras_input(keras_input, 0, train_size)
    valid_input = slice_keras_input(keras_input, train_size, train_size + valid_size)
    test_input = slice_keras_input(keras_input, train_size + valid_size, dataset.shape[0])

    hyper_params = {'batch_size': 2400,
                     'brand_name_embedding': 64,
                     'category1_embedding': 4,
                     'category2_embedding': 16,
                     'category3_embedding': 64,
                     'dense0': 610,
                     'dense1': 4,
                     'desc_embedding': 8,
                     'desc_filters': 32,
                     'dropout0': 0.41489440606260775,
                     'dropout1': 0.004588996677574161,
                     'lr0': 0.01100919733836959,
                     'lr1': 0.0004210931494001088,
                     'lr2': 0.0025172435657187654,
                     'name_embedding': 16,
                     'name_filters': 64}

    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4)) as sess:
        K.set_session(sess)

        model = build_cnn(hyper_params=hyper_params, **info_to_build_nn)

        lr_table = [hyper_params['lr0'], hyper_params['lr1'], hyper_params['lr2']]
        model.fit(
            train_input, train_y,
            validation_data=(valid_input, valid_y),
            batch_size=hyper_params['batch_size'],
            epochs=len(lr_table),
            callbacks=[LearningRateScheduler(lambda episode_i: lr_table[episode_i])]
        )

        valid_preds = model.predict(valid_input, batch_size=2048).reshape(-1)
        test_preds = model.predict(test_input, batch_size=2048).reshape(-1)

        del model
        gc.collect()

        print(rmse(valid_preds, valid_y))

    return {
        'valid_preds': valid_preds,
        'test_preds': test_preds
    }


def create_lgb_feature(features: List[Feature], start, end):
    lgb_features = []
    categorical_feature_indices = []

    index = 0
    for feature in features:
        gc.collect()
        lgb_feature = feature.get_lgb_features()[start:end]
        lgb_features.append(lgb_feature)

        if feature.get_feature_type() == FeatureType.CATEGORICAL:
            categorical_feature_indices.extend([index + i for i in range(lgb_feature.shape[1])])

        index += lgb_feature.shape[1]
    gc.collect()

    lgb_features = scipy.sparse.hstack(lgb_features, dtype=np.float64).tocsr()
    print('lgb_features.shape:', lgb_features.shape)
    gc.collect()

    # TODO: comment out
    assert lgb_features.shape[0] == end - start

    return lgb_features, categorical_feature_indices


def train_lgb_model(dataset, features: List[Feature]):
    timer = Timer('train_lgb_model')
    timer.print('Start')

    train_size = dataset['for_train'].sum()

    train_X, categorical_feature = create_lgb_feature(features, 0, train_size)
    train_y = dataset[:train_size]['log1p_price'].values
    d_train = lgb.Dataset(train_X, train_y, categorical_feature=categorical_feature)
    del train_X, train_y, categorical_feature
    gc.collect()
    timer.print('Created Dataset')

    params = {
        'application': 'regression',
        'metric': 'RMSE',
        'learning_rate': 0.3,
        # 'max_bin': 8192,
        'verbosity': -1,
        'seed': 114514,
        'nthread': 4
    }
    model = lgb.train(
        params,
        train_set=d_train,
        #         valid_sets=[d_train, d_valid],
        num_boost_round=900,
        # early_stopping_rounds=100,
        verbose_eval=100
    )
    gc.collect()
    timer.print('Trained')

    return model


def lgb_predict_test(dataset, features: List[Feature]):
    timer = Timer('lgb_predict_test')
    model = train_lgb_model(dataset, features)
    timer.print('Trained')

    train_size = dataset['for_train'].sum()
    test_size = dataset.shape[0] - train_size
    test_batches = 5
    test_batch_size = test_size // test_batches + 1

    test_preds = np.full(test_size, -114514, dtype=np.float64)
    for batch_i in range(test_batches):
        gc.collect()
        start = test_batch_size * batch_i
        end = min(test_size, test_batch_size * (batch_i + 1))
        test_X = create_lgb_feature(features, train_size + start, train_size + end)[0]
        test_preds[start:end] = model.predict(test_X)
    gc.collect()
    timer.print('Predicted test')

    return test_preds


def main():
    # price mean, different ngrams
    timer = Timer('main')
    timer.print('Start')

    # dataset = read_dataset(down_sampling_rate=0.001, train_size=0.9)
    dataset = read_dataset(train_size=0.999)  # train_size == 1だとvalid_size == 0で死ぬ
    timer.print('Loaded dataset')

    keras_result = do_keras(dataset)
    gc.collect()
    timer.print('Done keras')

    common_features = create_features(dataset)
    timer.print('Created features')

    feature_hyper_params = {
        'name_term': {
            'min_df': 1,
            'ngram_range': (1, 3),
        },
        'desc_term': {
            'min_df': 3,
            'ngram_range': (1, 2),
        },
    }
    sparse_features = create_sparse_features(dataset, feature_hyper_params)
    gc.collect()
    timer.print('Created sparse features')

    common_lgb_features = common_features + sparse_features
    common_lgb_features.append(create_ngram_price_mean(dataset, 'name', min_df=30, max_df=10000, ngram_range=(1, 3)))
    timer.print('Done create_ngram_price_mean')
    gc.collect()

    columns_to_keep = ['for_train', 'id', 'labeled', 'log1p_price', 'price']
    columns_to_drop = [c for c in dataset.columns if c not in columns_to_keep]
    for c in columns_to_drop:
        del dataset[c]
    gc.collect()

    common_lgb_features.append(create_ridge_predict_stacking_feature(dataset, common_features + sparse_features))
    gc.collect()
    timer.print('Done ridge predict')

    # common_lgb_features.append(create_fm_predict_stacking_feature(dataset, common_features + sparse_features))
    # timer.print('Done fm predict')

    lgb_test_preds_ = lgb_predict_test(dataset, common_lgb_features)
    timer.print('lgb Predicted test')

    train_size = dataset['for_train'].sum()
    valid_size = dataset['labeled'].sum() - train_size
    lgb_valid_preds = lgb_test_preds_[:valid_size]

    lgb_test_preds = lgb_test_preds_[valid_size:]

    ensemble_valid_preds = 0.7 * lgb_valid_preds + 0.3 * keras_result['valid_preds']
    ensemble_test_preds = 0.7 * lgb_test_preds + 0.3 * keras_result['test_preds']

    valid_train_y = dataset[train_size:train_size + valid_size]['log1p_price'].values
    print('lgb loss:', rmse(lgb_valid_preds, valid_train_y))
    print('CNN loss:', rmse(keras_result['valid_preds'], valid_train_y))
    print('Ensemble loss:', rmse(ensemble_valid_preds, valid_train_y))

    submission = pd.DataFrame()
    submission['test_id'] = dataset['id'][train_size + valid_size:]
    submission['price'] = np.expm1(ensemble_test_preds)     # TODO 3 clip
    submission.to_csv('submission.csv', index=False)
    timer.print('Saved submission.csv')

    return {
        'lgb_valid_preds': lgb_valid_preds,
        'keras_result': keras_result,
        'dataset': dataset,

        # 'ridge_feature': ridge_feature
    }


if __name__ == '__main__':
    main()

import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

valid_fold = 0

DATA_DIR = '../input/avito-demand-prediction/'
target_col = 'deal_probability'
#EMB_PATH = '../input/fasttest-common-crawl-russian/cc.ru.300.vec'
EMB_PATH = '../input/fasttext-russian-2m/wiki.ru.vec'

import sys
UTILS_PATH = '../input/myutils/'
if UTILS_PATH not in sys.path:
    sys.path.append(UTILS_PATH)

import adp_utils as au
eval_sets, y, train_num = au.load_eval_sets_lables()

from scipy import sparse
def get_ohe_data(col):
    path = '../input/adp-prepare-data-labelencoder-simple/data_lbe.csv'
    series = pd.read_csv(path, usecols=[col])[col]
    data = pd.get_dummies(series)
    data = data.iloc[:, :-1]
    del series; gc.collect();
    data = sparse.csr_matrix(data.values)
    return data

def get_num_data():
    z_score = lambda x: (x-x.mean())/x.std()
    data = au.get_common_feats()
    with au.timer('preprocess'):
        data.fillna(data.mean(), inplace=True)
        data = z_score(data)
    return data

def get_data():
    data = []
    cat_cols = [
        'region',
        'city',
        'parent_category_name',
        'category_name',
        'param_1','param_2','param_3',
        'user_type',
        'image','image_top_1'
    ]
    with au.timer('get cats & to sparse'):
        for col in cat_cols:
            data += [get_ohe_data(col)]
    with au.timer('get nums & to sparse'):
        data += [sparse.csr_matrix(get_num_data().values)]
    data = sparse.hstack(data, format='csr', dtype='float32')
    return data

import keras
import keras.backend as K
from keras import Model
from keras import optimizers
from keras.layers import Input, Dense, Dropout, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_error

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred-y_true)))

np.random.seed(233)  # for reproducibility

from keras.preprocessing import text, sequence
def get_coefs(word, *arr, tokenizer=None):
    if tokenizer is None:
        return word, np.asarray(arr, dtype='float32')
    else:
        if word not in tokenizer.word_index:
            return None
        else:
            return word, np.asarray(arr, dtype='float32')
def get_embedding_matrix(max_features, embed_size, tokenizer):
    embedding_matrix = np.zeros((max_features, embed_size))
    for o in open(EMB_PATH):
        res = get_coefs(*o.rstrip().rsplit(' '), tokenizer=tokenizer)
        if res is not None:
            idx = tokenizer.word_index[res[0]]
            if idx < max_features:
                embedding_matrix[idx] = res[1]
    return embedding_matrix
def get_emb_seq(max_features, embed_size, maxlen):
    df = pd.read_csv(
        '../input/adp-prepare-kfold-text/textdata.csv', 
        usecols=['context', 'text']
    )
    df['text'] = df['context'] + ' ' + df['text']
    del df['context']; gc.collect();
    tokenizer = text.Tokenizer(num_words=max_features)
    with au.timer('tokenizing...'):
        tokenizer.fit_on_texts(df['text'].values.tolist())
    max_features = min(max_features, len(tokenizer.word_index))
    with au.timer('getting embeddings'):
        embedding_matrix = get_embedding_matrix(max_features, embed_size, tokenizer)
    text_seq = tokenizer.texts_to_sequences(df['text'].values)
    del df; gc.collect();
    text_seq = sequence.pad_sequences(text_seq, maxlen=maxlen)
    del tokenizer; gc.collect();
    return embedding_matrix, text_seq, max_features

data = get_data()
print('data shape', data.shape)
n_samples, n_features = data.shape

max_features = 30000
embed_size = 300
maxlen = 127
with au.timer('get embed matrix & text seq'):
    embedding_matrix, text_seq, max_features = get_emb_seq(max_features, embed_size, maxlen)

from keras.layers import (
    BatchNormalization, SpatialDropout1D, GlobalAveragePooling1D, Embedding,
    GRU, Bidirectional, GlobalMaxPooling1D, Conv1D, CuDNNGRU
)

def get_model(
    seq_dim, tab_dim, 
    pre_fc_num, 
    num_fc_layers,
    rnn_params=[dict(units=u) for u in [64, 16]],
    lr=1e-3, dropout=0.25, decay=0.0
):
    seq_in = Input(shape=(seq_dim,), dtype='int32')
    text_emb = Embedding(max_features, 
                         embed_size, 
                         weights=[embedding_matrix],
                         name='text_emb')(seq_in)
    outs = []
    x_rnn = SpatialDropout1D(0.2)(text_emb)
    for i, rnnp in enumerate(rnn_params):
        x_rnn = CuDNNGRU(**rnnp, return_sequences=True)(x_rnn)
        x_rnn = BatchNormalization()(x_rnn)
        if i==len(rnn_params)-1:
            max_pool_rnn = GlobalMaxPooling1D()(x_rnn)
            avg_pool_rnn = GlobalAveragePooling1D()(x_rnn)
            outs += [avg_pool_rnn, max_pool_rnn]
    tab_in = Input(shape=(tab_dim,), dtype='float32', sparse=True)
    tab_out = Dense(pre_fc_num, activation='relu')(tab_in)
    x_out = concatenate([tab_out]+outs)
    for idx, num_fc in enumerate(num_fc_layers):
        x_out = Dense(num_fc, activation='relu')(x_out)
        if dropout>0:
            x_out = Dropout(dropout)(x_out)
    x_out = Dense(1)(x_out)
    model = Model([tab_in, seq_in], x_out)
    optimizer = optimizers.Adam(lr=lr, decay=decay)
    model.compile(loss=root_mean_squared_error, optimizer=optimizer)
    return model

def get_predict_res(params, epochs, X_train, y_train, X_valid, y_valid, X_test):
    model = get_model(**params)
    file_path = 'model.h5'
    check_point = ModelCheckpoint(file_path, 
                                  monitor='val_loss', 
                                  mode='min', 
                                  save_best_only=True, verbose=1) # 1
    scores = []
    for i in range(epochs):
        print('EPOCH TOTAL {} out of {}'.format(i+1, epochs))
        batch_size = min(2**13, 2**(10+i))
        hist = model.fit(X_train, y_train, 
                         validation_data=(X_valid, y_valid),
                         batch_size=batch_size, epochs=1, verbose=0,
                         callbacks=[check_point])
        scores.append(hist.history['val_loss'])
        if len(scores)>2 and scores[-1]>scores[-2]>scores[-3]:
            print('Early Stopped! Best score {}'.format(min(scores)))
            break
    model.load_weights(file_path)
    pred_val = model.predict(X_valid).ravel()
    pred_test = model.predict(X_test).ravel()
    return pred_val, pred_test, 0, 0, min(scores)

X = data[:train_num]
X_test = data[train_num:]
del data; gc.collect()
vec = text_seq[:train_num]
vec_test = text_seq[train_num:]
del text_seq; gc.collect()

n_final_state = 32
scores = []
pred_test_all = np.zeros((X_test.shape[0],))
epochs = 20
params = dict(tab_dim=X_test.shape[1], seq_dim=maxlen)
params['pre_fc_num'] = 128
params['num_fc_layers'] = [64, n_final_state]
params['rnn_params'] = [dict(units=u) for u in [64, 16]]
params['dropout'] = -1#0.125
params['lr'] = 9e-4
params['decay'] = 0.0

def get_train_test_split(Xs, y, 
                         valid_fold=valid_fold, eval_sets=eval_sets):
    mask_val = eval_sets==valid_fold
    mask_tr = ~mask_val
    X_train = [X[mask_tr] for X in Xs]
    y_train = y[mask_tr]
    X_valid = [X[mask_val] for X in Xs]
    y_valid = y[mask_val]
    return X_train, X_valid, y_train, y_valid

X_test = [X_test, vec_test]
for valid_fold in range(10):
    with au.timer('FOLD {}'.format(valid_fold+1)):
        X_train, X_valid, y_train, y_valid = get_train_test_split(
            [X, vec], y, valid_fold=valid_fold)
        pred_val, pred_test, valid_state, test_state, score = get_predict_res(
            params, epochs, X_train, y_train, X_valid, y_valid, X_test)
    scores.append(score)
    del X_train, y_train, X_valid, y_valid; gc.collect();
    np.save('valid_%d_pred.npy'%valid_fold, pred_val)
    pred_test_all += pred_test/10
np.save('test_pred.npy', pred_test_all)

sub = pd.read_csv(DATA_DIR+'sample_submission.csv')
sub[target_col] = pred_test_all
sub.to_csv('mlp_{}.csv'.format(np.mean(scores)), index=False)
print('save to', 'mlp_{}.csv'.format(np.mean(scores)))









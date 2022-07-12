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

import sys
UTILS_PATH = '../input/myutils/'
if UTILS_PATH not in sys.path:
    sys.path.append(UTILS_PATH)

import adp_utils as au
eval_sets, y, train_num = au.load_eval_sets_lables()

from scipy import sparse
def get_ohe_data(col):
    path = '../input/adp-prepare-data-labelencoder-simple/data_lbe.csv'
    with au.timer('load {}'.format(col)):
        series = pd.read_csv(path, usecols=[col])[col]
    with au.timer('to one-hot'):
        data = pd.get_dummies(series)
        data = data.iloc[:, :-1]
        del series; gc.collect();
    with au.timer('to sparse'):
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
        'region','city',
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
    with au.timer('get text vec'):
        data += [au.get_text_vec('text_all', 'all')]
    data = sparse.hstack(data, format='csr', dtype='float32')
    return data

def get_train_test_split(X, y, 
                         valid_fold=valid_fold, eval_sets=eval_sets):
    mask_val = eval_sets==valid_fold
    mask_tr = ~mask_val
    X_train = X[mask_tr]
    y_train = y[mask_tr]
    X_valid = X[mask_val]
    y_valid = y[mask_val]
    return X_train, X_valid, y_train, y_valid

import keras
import keras.backend as K
from keras import Model
from keras import optimizers
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_error

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred-y_true)))

def get_model(input_dim,
              num_fc_layers,
              lr=1e-3,
              dropout=0.25,
              decay=0.0):
    x_in = Input(shape=(input_dim,), dtype='float32', sparse=True)
    x_out = x_in
    for idx, num_fc in enumerate(num_fc_layers):
        x_out = Dense(num_fc, activation='relu')(x_out)
        if dropout>0:
            x_out = Dropout(dropout)(x_out)
    x_out = Dense(1)(x_out)
    model = Model(x_in, x_out)
    optimizer = optimizers.Adam(lr=lr, decay=decay)
    model.compile(loss=root_mean_squared_error, optimizer=optimizer)
    return model

def get_predict_res(params, epochs, X_train, y_train, X_valid, y_valid, X_test):
    model = get_model(**params)
    file_path = 'model.h5'
    check_point = ModelCheckpoint(file_path, 
                                  monitor='val_loss', 
                                  mode='min', 
                                  save_best_only=True, verbose=1)
    scores = []
    for i in range(epochs):
        print('EPOCH TOTAL {} out of {}'.format(i+1, epochs))
        batch_size = min(2**13, 2**(10+i))
        hist = model.fit(X_train, y_train, 
                         validation_data=(X_valid, y_valid),
                         batch_size=batch_size, epochs=1, verbose=1,
                         callbacks=[check_point])
        scores.append(hist.history['val_loss'])
        if len(scores)>2 and scores[-1]>scores[-2]>scores[-3]:
            print('Early Stopped! Best score {}'.format(min(scores)))
            break
    model.load_weights(file_path)
    pred_val = model.predict(X_valid).ravel()
    pred_test = model.predict(X_test).ravel()
    model_state = Model(model.input, model.get_layer(index=-2).output)
    valid_state =  model_state.predict(X_valid)
    test_state = model_state.predict(X_test)
    return pred_val, pred_test, valid_state, test_state, min(scores)

data = get_data()
X = data[:train_num]
X_test = data[train_num:]
del data; gc.collect()

n_features = X_test.shape[1]
n_final_state = 16
scores = []
scoresb = []
pred_test_all = np.zeros((X_test.shape[0],))
predb_test_all = np.zeros((X_test.shape[0],))
test_state_all = np.zeros((X_test.shape[0], n_final_state))
testb_state_all = np.zeros((X_test.shape[0], n_final_state))
epochs = 20
params = dict(input_dim=n_features)
params['num_fc_layers'] = [128, n_final_state]
params['dropout'] = -1#0.125
params['lr'] = 1e-4
params['decay'] = 0.0
for valid_fold in range(10):
    print('TOTAL FOLD {}'.format(valid_fold+1))
    X_train, X_valid, y_train, y_valid = get_train_test_split(
        X, y, valid_fold=valid_fold)
    Xb_train, Xb_valid, Xb_test = [x.astype('bool').astype('float32') \
                          for x in [X_train, X_valid, X_test]]
    pred_val, pred_test, valid_state, test_state, score = get_predict_res(
        params, epochs, X_train, y_train, X_valid, y_valid, X_test)
    scores.append(score)
    np.save('valid_%d_pred.npy'%valid_fold, pred_val)
    np.savez('valid_%d_state.npz'%valid_fold, valid_state)
    pred_test_all += pred_test/10
    test_state_all += test_state/10
    predb_val, predb_test, validb_state, testb_state, scoreb = get_predict_res(
        params, epochs, Xb_train, y_train, Xb_valid, y_valid, Xb_test)
    scoresb.append(scoreb)
    np.save('validb_%d_pred.npy'%valid_fold, predb_val)
    np.savez('validb_%d_state.npz'%valid_fold, validb_state)
    predb_test_all += predb_test/10
    testb_state_all += testb_state/10
np.save('test_pred.npy', pred_test_all)
np.savez('test_state.npz', test_state_all)
np.save('testb_pred.npy', predb_test_all)
np.savez('testb_state.npz', testb_state_all)

sub = pd.read_csv(DATA_DIR+'sample_submission.csv')
sub[target_col] = pred_test_all
sub.to_csv('mlp_{}.csv'.format(np.mean(scores)), index=False)
print('save to', 'mlp_{}.csv'.format(np.mean(scores)))

sub = pd.read_csv(DATA_DIR+'sample_submission.csv')
sub[target_col] = predb_test_all
sub.to_csv('mlpb_{}.csv'.format(np.mean(scoresb)), index=False)
print('save to', 'mlpb_{}.csv'.format(np.mean(scoresb)))






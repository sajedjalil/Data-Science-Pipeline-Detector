# -*- coding: utf-8 -*-
import os
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, preprocessing
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from keras.layers import Dense, Input,Multiply,Add,Concatenate
#from collections import Counter

from keras.layers import BatchNormalization 
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras import callbacks
from keras import backend as K
from keras.layers import Dropout


#import warnings
#warnings.filterwarnings("ignore")


def mut(arr):
    res = 1
    for a in arr:
        res *= a
    return res

def sparse_cont_feature(data,k,pre):
    w = [1.0*i/k for i in range(k+1)]
    w = data.describe(percentiles=w)[4:4+k+1]
    w[0] = w[0]*(1-1e-10)
    dt_id = pd.cut(data, w, labels=range(k))
    dt_sparse = pd.get_dummies(dt_id,prefix = pre)
    return dt_sparse


def submit(predictions):
    submit = pd.read_csv('../input/sample_submission.csv')
    submit["target"] = predictions
    submit.to_csv("submission.csv", index=False)

def fallback_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except:
        return 0.5

def auc(y_true, y_pred):
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)

df_tr = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


useful = np.zeros((256,512))

for i in range(512):
    partial = df_tr[ df_tr['wheezy-copper-turtle-magic']==i ]
    useful[:,i] = np.std(partial.iloc[:,1:-1], axis=0)
# CONVERT TO BOOLEANS IDENTIFYING USEFULNESS
useful = useful > 1.5
useful[146,:] = [True]*512
for i in range(512):
    idx = df_tr.columns[1:-1][ ~useful[:,i] ]    
    df_tr.loc[ df_tr.iloc[:,147]==i,idx ] = 0.0
    df_test.loc[ df_test.iloc[:,147]==i,idx ] = 0.0 
    
NFOLDS = 10
RANDOM_STATE = 42
numeric = [c for c in df_tr.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]


len_train = df_tr.shape[0]
df_test['target'] = -1
data = pd.concat([df_tr, df_test])

data['magic_count'] = data.groupby(['wheezy-copper-turtle-magic'])['id'].transform('count')
data = pd.concat([data, pd.get_dummies(data['wheezy-copper-turtle-magic'],prefix = 'wheezy-copper-turtle-magic')], axis=1, sort=False)
# data = data.drop(['wheezy-copper-turtle-magic'],axis=1)
df_tr = data[:len_train]
df_test = data[len_train:]


folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_STATE)


gc.collect()


y = df_tr.target
ids = df_tr.id.values
train = df_tr.drop(['id', 'target'], axis=1)
test_ids = df_test.id.values
test = df_test[train.columns]

oof_preds = np.zeros((len(train)))
test_preds = np.zeros((len(test)))

scl = preprocessing.StandardScaler()
scl.fit(pd.concat([train, test]))

train = scl.transform(train)
test = scl.transform(test)

for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
    print("Current Fold: {}".format(fold_))
    trn_x, trn_y = train[trn_, :], y.iloc[trn_]
    val_x, val_y = train[val_, :], y.iloc[val_]
    
    inp = Input(shape=(trn_x.shape[1],))
    inp = Input(shape=(256,))
    inp1 = Input(shape=(512,))
    z  = Dense(512, activation="relu")(inp1)
    z = BatchNormalization()(z)
    z = Dropout(0.1)(z)
    x1 = Dense(10000, activation="relu")(inp)
    x2 = BatchNormalization()(x1)
    x3 = Dropout(0.3)(x2)
    x4 = Dense(5000, activation="relu")(x3)
    x5 = BatchNormalization()(x4)
    x6 = Dropout(0.3)(x5)
    x7 = Dense(512, activation="relu")(x6)
    x8 = BatchNormalization()(x7)
    x9 = Dropout(0.2)(x8)
    out = Dense(1, activation="sigmoid")(Concatenate()([x9,Multiply()([x9,z])]))
    clf = Model(inputs=[inp,inp1], outputs=out)
    clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc])

    es = callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=15,
                                 verbose=1, mode='max', baseline=None, restore_best_weights=True)

    rlr = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                                      patience=3, min_lr=1e-6, mode='max', verbose=1)

    clf.fit([trn_x[:,0:256],trn_x[:,256:768]], trn_y, validation_data=([val_x[:,0:256],val_x[:,256:768]], val_y), callbacks=[es, rlr], epochs=100, batch_size=1024, verbose=2)
    
    val_preds = clf.predict([val_x[:,0:256],val_x[:,256:768]])
    test_fold_preds = clf.predict([test[:,0:256],test[:,256:768]])
    
    print("AUC = {}".format(metrics.roc_auc_score(val_y, val_preds)))
    oof_preds[val_] = val_preds.ravel()
    test_preds += test_fold_preds.ravel() / NFOLDS
    
    K.clear_session()
    gc.collect()
    
submit(test_preds)
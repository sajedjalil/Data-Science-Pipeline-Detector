from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score


'''
    This demonstrates how to run a Keras Deep Learning model for ROC AUC score (local 4-fold validation)
    for the springleaf challenge

    The model trains in a few seconds on CPU.
'''


def float32(k):
    return np.cast['float32'](k)
    
def loss_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)
    
def load_train_data(path):
    print("Loading Train Data")
    df = pd.read_csv(path)
    
    
    # Remove line below to run locally - Be careful you need more than 8GB RAM 
    df = df.sample(n=60000)
    
    labels = df.target

    df = df.drop('target',1)
    df = df.drop('ID',1)
    
    # Junk cols - Some feature engineering needed here
    df = df.ix[:, 520:660].fillna(-1)

    X = df.values.copy()
    
    np.random.shuffle(X)

    X = X.astype(np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler

def load_test_data(path, scaler):
    print("Loading Test Data")
    df = pd.read_csv(path)
    ids = df.ID.astype(str)

    df = df.drop('ID',1)
    
    # Junk cols - Some feature engineering needed here
    df = df.ix[:, 520:660].fillna(-1)
    X = df.values.copy()

    X, = X.astype(np.float32),
    X = scaler.transform(X)
    return X, ids
    

def build_model(input_dim, output_dim):
    print("building model")
    N1=256#128
    N2=256#128
    N3=256#128
    model = Sequential()
    
    model.add(Dense(input_dim, N1, init='glorot_uniform'))
    model.add(PReLU((N1,)))
    model.add(BatchNormalization((N1,)))
    model.add(Dropout(0.5))

    model.add(Dense(N1, N2, init='glorot_uniform'))
    model.add(PReLU((N2,)))
    model.add(BatchNormalization((N2,)))
    model.add(Dropout(0.5))
    
    model.add(Dense(N2, N3, init='glorot_uniform'))
    model.add(PReLU((N3,)))
    model.add(BatchNormalization((N3,)))
    model.add(Dropout(0.5))

    model.add(Dense(N3, output_dim, init='glorot_uniform'))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer="adadelta")
    return model


if __name__ == "__main__":
    # Load data set and target values

    print("Loading train data")
    X, y, encoder, scaler = load_train_data("../input/train.csv")
    '''Convert class vector to binary class matrix, for use with categorical_crossentropy'''
    Y = np_utils.to_categorical(y)

    print("Loading train data")
    X_test, ids = load_test_data("../input/test.csv", scaler)
    print('Number of classes:', len(encoder.classes_))

    input_dim = X.shape[1]
    output_dim = len(encoder.classes_)


    print("Generating submission...")

    model = build_model(input_dim, output_dim)
    model.fit(X, Y, nb_epoch=10, batch_size=64, verbose=1)#10

    preds = model.predict_proba(X_test, verbose=0)[:, 1]
    submission = pd.DataFrame(preds, index=ids, columns=['target'])
    submission.to_csv('Keras_deep_256_ssz.csv')    

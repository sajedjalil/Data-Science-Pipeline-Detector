from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from theano import shared

from keras.utils import np_utils
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax, sigmoid
from lasagne.objectives import binary_crossentropy
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit

def float32(k):
    return np.cast['float32'](k)
    
def load_train_data(path):
    print("Loading Train Data")
    df = pd.read_csv(path)
    
    
    # Remove line below to run locally - Be careful you need more than 8GB RAM 
    df = df.sample(n=40000)
    
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

if __name__ == "__main__":
    # Load data set and target values
    X, y, encoder, scaler = load_train_data("../input/train.csv")
    Y = np_utils.to_categorical(y)
    X_test, ids = load_test_data("../input/test.csv", scaler)
    print('Number of classes:', len(encoder.classes_))
    num_classes = len(encoder.classes_)
    num_features = X.shape[1]


    layers0 = [('input', InputLayer),
               ('dense0', DenseLayer),
               ('dropout0', DropoutLayer),
               ('dense1', DenseLayer),
               ('output', DenseLayer)]

    net0 = NeuralNet(layers=layers0,

                     input_shape=(None, num_features),
                     dense0_num_units=100,
                     dropout0_p=0.1,
                     dense1_num_units=100,

                     output_num_units=num_classes,
                     output_nonlinearity=sigmoid,

                     update=nesterov_momentum,
                     update_learning_rate=0.3,
                     update_momentum=0.8,
                     
                     objective_loss_function = binary_crossentropy,
                     
                     train_split=TrainSplit(0.1),
                     verbose=1,
                     max_epochs=20)

    net0.fit(X, Y)
    print('Prediction Complete')
    preds = net0.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame(preds, index=ids, columns=['target'])
    submission.to_csv('BTB_Lasagne.csv')
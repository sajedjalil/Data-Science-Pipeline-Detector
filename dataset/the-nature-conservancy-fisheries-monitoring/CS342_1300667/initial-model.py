from __future__ import print_function
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


import math
#import time
#import warnings
#from skimage import data, io, filters
import cv2
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score as cv
from sklearn.model_selection import cross_val_predict
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt


def load_dataset():
    url = '/train/'

def load_train_data():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('..', 'input', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id
  
def load_test_data():
    path = os.path.join('..')
    
    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id

#create the CSV file ready for transmission with all the predictions

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)
 
 
   
def read_and_normalize_train_data():
    train_data, train_target, train_id = load_train_data()

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, 8)

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id

def create_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 32, 32), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model

#----------------

def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

def predictTNA(df,tLabel):
    mlp = MLPRegressor(hidden_layer_sizes=(10,10)) #other parameters -- ,activation = 'logistic')
    Y = df[tLabel]
    X = df.drop(tLabel,axis=1).copy()

    #k_fold = KFold(n_splits=10)
    #scores = []
    #for train,test in k_fold.split(X):
    #    mlp = mlp.fit(X.iloc[train],Y.iloc[train])
    #    s = mlp.score(X.iloc[test],Y.iloc[test])
    #    scores.append(s)
    #print scores
    predicted = cross_val_predict(mlp,X,Y,cv=10)
    print (r2_score(Y,predicted))


def predictClass(df,tLabel):
    mlp = MLPClassifier(hidden_layer_sizes=(10,10))
    Y = df[tLabel]
    X = df.drop(tLabel,axis=1).copy()
    predicted = cross_val_predict(mlp,X,Y,cv=10)
    print (roc_auc_score(Y,predicted))

def predictKerasClass(df,tLabel):
    Y = df[tLabel]
    X = df.drop(tLabel,axis=1).copy()
    
    #create model with 2 hidden layers of 10 units, and an output layer
    model = Sequential()
    model.add(Dense(10,input_dim=13,init='uniform',activation='relu'))
    model.add(Dropout(0.2)) #this randomly sets some input units to 0 - prevents overfitting
    model.add(Dense(10,init='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(1,init='normal'))

    model.compile(loss='mean_squared_error',optimizer='adam')

    k_fold = KFold(n_splits=10)
    scores = []
    #just one iteration of kFold CV
    for train,test in k_fold.split(X):
        model.fit(X.iloc[train].as_matrix(),Y.iloc[train].as_matrix(),nb_epoch=20, batch_size=100)
        s = model.evaluate(X.iloc[test].as_matrix(),Y.iloc[test].as_matrix(),batch_size=100)
        scores.append(s)
        break
    print (scores)

    
    predictKerasClass(url, '/test_stg1')
    
    


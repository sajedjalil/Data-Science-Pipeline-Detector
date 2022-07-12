# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

import time
import sys
import gc

def run_solution_func():
    #reading data
    print('Reading training data')
    df_train = pd.read_csv('../input/train.csv', dtype=dict(row_id=np.int32, x=np.float32, y=np.float32, accuracy=np.int32, time=np.int32, place_id=np.int64))
    X_train = df_train.drop(['row_id', 'place_id'], axis=1)
    y_train = df_train['place_id']
    
    print('Reading testing data')
    df_test = pd.read_csv('../input/test.csv', dtype=dict(row_id=np.int32, x=np.float32, y=np.float32, accuracy=np.int32, time=np.int32))
    X_test = df_test.drop(['row_id'], axis=1)
    id_test = df_test['row_id']
    
    #training model
    print('\nTraining model')
    start_time = time.time()
    neigh = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='kd_tree')
    neigh.fit(X_train, y_train)
    m, s = divmod(time.time() - start_time, 60)
    h, m = divmod(m, 60)
    print('Cost %02d hrs: %02d mins: %02d secs to train model' % (h, m, s))

    #predicting
    print('\nPredicting')
    start_time = time.time()
    res = neigh.predict(X_test)
    m, s = divmod(time.time() - start_time, 60)
    h, m = divmod(m, 60)
    print('Cost %02d hrs: %02d mins: %02d secs to predict' % (h, m, s))

    submission = pd.DataFrame({"row_id":id_test, "place_id":res})
    submission.to_csv('submission_naive_knn', index=False)

if __name__ == '__main__':
    run_solution_func()
